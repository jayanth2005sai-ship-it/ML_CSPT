"""
house_price_app.py
--------------------
All-in-one House Price Prediction app using Linear Regression + Streamlit.

This single file:
  1. Generates a synthetic housing dataset (cached, so it's created once).
  2. Trains a Linear Regression model on it (cached).
  3. Renders an interactive Streamlit UI for making predictions and
     exploring the data.

Run with:
    streamlit run house_price_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

FEATURE_COLS = [
    "area_sqft", "bedrooms", "bathrooms",
    "stories", "age_years", "garage_spaces", "distance_to_city_km",
]


# ---------------------------------------------------------------
# Data generation (cached so it only runs once per session)
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

    return pd.DataFrame({
        "area_sqft": area.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "age_years": age,
        "garage_spaces": garage,
        "distance_to_city_km": distance_to_city.round(2),
        "price": price.round(0),
    })


# ---------------------------------------------------------------
# Model training (cached so it only trains once per session)
# ---------------------------------------------------------------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLS]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    return model, scaler, metrics


# ---------------------------------------------------------------
# Load data + train model
# ---------------------------------------------------------------
data = generate_dataset()
model, scaler, metrics = train_model(data)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "This app uses a **Linear Regression** model to estimate a house's "
    "price based on its features. Adjust the inputs in the sidebar and "
    "click **Predict** to see the estimated price."
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        f"""
        - **Algorithm:** Linear Regression (scikit-learn)
        - **Test MAE:** ${metrics['mae']:,.0f}
        - **Test RMSE:** ${metrics['rmse']:,.0f}
        - **Test R² score:** {metrics['r2']:.3f}

        The dataset used here is synthetically generated for demonstration
        purposes. Replace `generate_dataset()` with a real CSV load to use
        this app with real-world data.
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

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.success(f"### Estimated Price: **${prediction:,.0f}**")

    lower = prediction - metrics["rmse"]
    upper = prediction + metrics["rmse"]
    st.caption(f"Likely range: ${lower:,.0f} – ${upper:,.0f} (±1 RMSE)")

st.divider()

# ---------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------
st.subheader("📊 Explore the Training Data")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Houses in Dataset", len(data))
with col2:
    st.metric("Average Price", f"${data['price'].mean():,.0f}")

tab1, tab2, tab3 = st.tabs(["Price vs Area", "Price Distribution", "Raw Data"])

with tab1:
    st.scatter_chart(data, x="area_sqft", y="price")

with tab2:
    st.bar_chart(data["price"].value_counts(bins=20).sort_index())

with tab3:
    st.dataframe(data, use_container_width=True)

# ---------------------------------------------------------------
# Feature importance (coefficients)
# ---------------------------------------------------------------
st.subheader("🔍 Feature Impact on Price")
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": model.coef_,
}).sort_values("coefficient", key=abs, ascending=False)

st.bar_chart(coef_df.set_index("feature")["coefficient"])
st.caption(
    "Coefficients are based on scaled features, so they reflect each "
    "feature's relative influence on the predicted price."
)
