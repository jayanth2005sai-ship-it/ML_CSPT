"""
house_price_logistic_app.py
-----------------------------
House Price Classification using Logistic Regression + Streamlit.

Unlike Linear Regression (which predicts an exact price number), Logistic
Regression predicts a CATEGORY. Here we classify a house as:
    0 = "Affordable"  (price below the dataset median)
    1 = "Expensive"   (price at/above the dataset median)

This single file:
  1. Generates a synthetic housing dataset (cached).
  2. Creates a binary label from price (Affordable / Expensive).
  3. Trains a Logistic Regression classifier (cached).
  4. Renders an interactive Streamlit UI for predictions + exploration.

Run with:
    streamlit run house_price_logistic_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(page_title="House Price Classifier", page_icon="🏠", layout="centered")

FEATURE_COLS = [
    "area_sqft", "bedrooms", "bathrooms",
    "stories", "age_years", "garage_spaces", "distance_to_city_km",
]


# ---------------------------------------------------------------
# Data generation (cached)
# ---------------------------------------------------------------
@st.cache_data
def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    area = rng.normal(1800, 650, n_samples).clip(300, 6000)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples)
    stories = rng.integers(1, 4, n_samples)
    age = rng.integers(0, 50, n_samples)
    garage = rng.integers(0, 3, n_samples)
    distance_to_city = rng.uniform(0.5, 30, n_samples)

    price = (
        area * 120
        + bedrooms * 8000
        + bathrooms * 6000
        + stories * 4000
        + garage * 5000
        - age * 500
        - distance_to_city * 900
        + rng.normal(0, 15000, n_samples)
        + 25000
    )
    price = price.clip(30000, None)

    df = pd.DataFrame({
        "area_sqft": area.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "age_years": age,
        "garage_spaces": garage,
        "distance_to_city_km": distance_to_city.round(2),
        "price": price.round(0),
    })

    # Binary target: 1 = Expensive (>= median price), 0 = Affordable
    median_price = df["price"].median()
    df["is_expensive"] = (df["price"] >= median_price).astype(int)
    return df, median_price


# ---------------------------------------------------------------
# Model training (cached)
# ---------------------------------------------------------------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLS]
    y = data["is_expensive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return model, scaler, metrics


# ---------------------------------------------------------------
# Load data + train model
# ---------------------------------------------------------------
data, median_price = generate_dataset()
model, scaler, metrics = train_model(data)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🏠 House Price Classifier")
st.write(
    "This app uses **Logistic Regression** to classify a house as "
    "**Affordable** or **Expensive** based on its features, relative to "
    f"the median price of **${median_price:,.0f}** in the dataset."
)

with st.expander("ℹ️ About this model"):
    cm = metrics["confusion_matrix"]
    st.markdown(
        f"""
        - **Algorithm:** Logistic Regression (scikit-learn)
        - **Accuracy:** {metrics['accuracy']:.2%}
        - **Precision:** {metrics['precision']:.2%}
        - **Recall:** {metrics['recall']:.2%}
        - **F1 Score:** {metrics['f1']:.2%}

        **Confusion Matrix** (rows = actual, columns = predicted):

        |                  | Predicted Affordable | Predicted Expensive |
        |------------------|----------------------|----------------------|
        | **Actual Affordable** | {cm[0][0]} | {cm[0][1]} |
        | **Actual Expensive**  | {cm[1][0]} | {cm[1][1]} |

        The dataset is synthetically generated for demonstration. Swap in
        real housing data (with a `price` column) to use this with
        real-world data.
        """
    )

# ---------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area (sq ft)", 300, 6000, 1800, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
stories = st.sidebar.slider("Stories", 1, 4, 1)
age = st.sidebar.slider("Age of house (years)", 0, 50, 10)
garage = st.sidebar.slider("Garage spaces", 0, 3, 1)
distance = st.sidebar.slider("Distance to city center (km)", 0.5, 30.0, 10.0, step=0.5)

input_df = pd.DataFrame([{
    "area_sqft": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "age_years": age,
    "garage_spaces": garage,
    "distance_to_city_km": distance,
}])[FEATURE_COLS]

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.subheader("Your Input")
st.dataframe(input_df, hide_index=True, use_container_width=True)

if st.button("🔮 Classify House", type="primary", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    label = "Expensive 💰" if prediction == 1 else "Affordable 🏡"
    confidence = probabilities[prediction]

    if prediction == 1:
        st.success(f"### Prediction: **{label}**")
    else:
        st.info(f"### Prediction: **{label}**")

    st.caption(f"Confidence: {confidence:.1%}")

    prob_df = pd.DataFrame({
        "Category": ["Affordable", "Expensive"],
        "Probability": probabilities,
    }).set_index("Category")
    st.bar_chart(prob_df)

st.divider()

# ---------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------
st.subheader("📊 Explore the Training Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Houses", len(data))
with col2:
    st.metric("Median Price", f"${median_price:,.0f}")
with col3:
    st.metric("Expensive Houses", int(data["is_expensive"].sum()))

tab1, tab2, tab3 = st.tabs(["Area vs Price (by Class)", "Class Balance", "Raw Data"])

with tab1:
    chart_data = data.copy()
    chart_data["Class"] = chart_data["is_expensive"].map({0: "Affordable", 1: "Expensive"})
    st.scatter_chart(chart_data, x="area_sqft", y="price", color="Class")

with tab2:
    counts = data["is_expensive"].map({0: "Affordable", 1: "Expensive"}).value_counts()
    st.bar_chart(counts)

with tab3:
    st.dataframe(data, use_container_width=True)

# ---------------------------------------------------------------
# Feature importance (coefficients)
# ---------------------------------------------------------------
st.subheader("🔍 Feature Impact on Classification")
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

st.bar_chart(coef_df.set_index("feature")["coefficient"])
st.caption(
    "Positive coefficients push the prediction toward 'Expensive'; "
    "negative coefficients push toward 'Affordable'. Based on scaled features."
)
