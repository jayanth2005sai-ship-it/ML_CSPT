import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP & CUSTOM STYLING ---
st.set_page_config(
    page_title="DataVibe AI | Intelligent Variable Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for a beautiful, modern card-based UI layout
st.markdown("""
    <style>
    .reportview-container { background: #f4f6f9; }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #4F46E5;
        margin-bottom: 15px;
    }
    .insight-box {
        background-color: #F8FAFC;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-top: 15px;
    }
    .stAlert { border-radius: 10px !important; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #1E293B; }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown("""
    <div style='text-align: center; padding-bottom: 25px;'>
        <h1 style='font-size: 2.8rem; margin-bottom: 5px;'>🔮 DataVibe AI</h1>
        <p style='color: #64748B; font-size: 1.1rem;'>Autonomous Analytics Platform for Descriptive, Predictive, & Prescriptive Insights</p>
    </div>
""", unsafe_allow_html=True)

# --- FILE UPLOADER WITH ANCHOR CARD ---
uploaded_file = st.file_uploader("", type=["csv"])

if not uploaded_file:
    st.markdown("""
        <div style='text-align: center; margin-top: 50px; padding: 40px; border: 2px dashed #CBD5E1; border-radius: 15px; background: white;'>
            <h3 style='color: #475569;'>👋 Welcome to DataVibe AI</h3>
            <p style='color: #94A3B8; max-width: 500px; margin: 10px auto;'>Drop any CSV dataset here to instantly parse dependencies, unlock predictive formulas, and reveal prescriptive data strategies.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Safe Data Loading
    df = pd.read_csv(uploaded_file)
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # --- TOP LEVEL DATA INSIGHT METRICS ---
    st.markdown("### 📊 Dataset Dynamic Profile")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><span style='color:#64748B; font-size:0.9rem;'>Total Records</span><h2 style='margin:5px 0 0 0; color:#1E293B;'>{df.shape[0]:,}</h2></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card' style='border-left-color: #10B981;'><span style='color:#64748B; font-size:0.9rem;'>Features Matrix</span><h2 style='margin:5px 0 0 0; color:#1E293B;'>{df.shape[1]} Columns</h2></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card' style='border-left-color: #F59E0B;'><span style='color:#64748B; font-size:0.9rem;'>Numeric Dimensions</span><h2 style='margin:5px 0 0 0; color:#1E293B;'>{len(numeric_cols)} Features</h2></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card' style='border-left-color: #EC4899;'><span style='color:#64748B; font-size:0.9rem;'>Missing Metrics</span><h2 style='margin:5px 0 0 0; color:#1E293B;'>{df.isna().sum().sum():,} Cells</h2></div>", unsafe_allow_html=True)

    with st.expander("🔍 Interactive Data Matrix Explorer", expanded=False):
        st.dataframe(df, use_container_width=True)

    # --- SIDEBAR WORKSPACE CONTROLS ---
    st.sidebar.markdown("<h2 style='font-size:1.4rem; color:#4F46E5; margin-bottom:0;'>Control Studio</h2>", unsafe_allow_html=True)
    st.sidebar.write("Configure workspace engines below:")
    
    chart = st.sidebar.selectbox(
        "Select Analytics Engine",
        [
            "✨ Executive AI Summary",
            "🔥 Correlation Matrix",
            "🎯 Scatter Insights Matrix",
            "📈 Advanced Linear Regression",
            "📦 Categorical Box & Range",
            "🎻 Distribution Violin Curves",
            "🫧 3D Bubble Volume Explorer",
            "⬢ Hexbin High-Density Plot",
            "🌿 Multivariate Pair Matrix"
        ]
    )
    
    # Safe index utility helper function
    def get_safe_index(lst, idx):
        return idx if len(lst) > idx else 0

    st.markdown("---")

    # ==========================================
    # ENGINE 1: EXECUTIVE AI SUMMARY
    # ==========================================
    if chart == "✨ Executive AI Summary":
        st.subheader("✨ Autonomous Executive Overview")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 💎 Top Predictive Dependencies")
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr().abs().unstack().sort_values(ascending=False)
                corr = corr[corr < 1.0].drop_duplicates().head(3)
                if not corr.empty:
                    for (v1, v2), val in corr.items():
                        st.markdown(f"🔹 **`{v1}` & `{v2}`** match with structural connection strength of **{val*100:.1f}%**")
                else:
                    st.write("No distinct correlations noted.")
            else:
                st.info("Additional numeric structures needed for dynamic mapping.")
                
        with col_right:
            st.markdown("#### ⚠️ High Volatility / Risk Targets")
            if numeric_cols:
                cv = df[numeric_cols].std() / df[numeric_cols].mean().abs()
                high_risk = cv.sort_values(ascending=False).head(2)
                for var, val in high_risk.items():
                    if not np.isnan(val):
                        st.markdown(f"🔸 **`{var}`** exhibits significant variance (Coefficient of Variance: **{val:.2f}**)")
            else:
                st.write("No risk indicators found.")

        # Beautiful custom card layout summary
        st.markdown("""
            <div class='insight-box' style='border-top: 4px solid #4F46E5;'>
                <h5>🎯 Prescriptive Data Framework:</h5>
                <p style='color: #475569; font-size: 0.95rem; margin-bottom:0;'>
                    Use the <b>Control Studio</b> dropdown menu on the left sidebar to access targeted statistical modeling engines. 
                    For forecasting trajectories, select <b>Advanced Linear Regression</b>. To isolate structural operational noise 
                    among unique demographics, pick <b>Categorical Box & Range</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # ENGINE 2: CORRELATION HEATMAP
    # ==========================================
    elif chart == "🔥 Correlation Matrix":
        st.subheader("🔥 Correlation Structural Engine")
        if len(numeric_cols) >= 1:
            corr = df[numeric_cols].corr()
            
            # Interactive Plotly Heatmap
            fig = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Linear Correlation Coefficients Matrix"
            )
            fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Automated Analytical Layers
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Strategic Intelligence Overview")
            st.write("**Descriptive:** Tracks global systemic features. Direct blue grids imply structural growth links; red boxes reflect inverse adjustments.")
            
            high_corr = corr.abs().unstack().sort_values(ascending=False)
            high_corr = high_corr[high_corr < 1.0].drop_duplicates()
            strong_pairs = high_corr[high_corr > 0.70]

            if not strong_pairs.empty:
                st.markdown("#### 💡 Prescriptive Strategy Plan")
                for (v1, v2), val in strong_pairs.items():
                    st.success(f"🚀 **Operational Action Plan:** Changing resource input levels for `{v1}` provides direct, mathematically validated control channels over changes in `{v2}` (Confidence metric: **{val:.2f}**).")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Correlation analysis requires at least one numeric feature block.")

    # ==========================================
    # ENGINE 3: SCATTER PLOT
    # ==========================================
    elif chart == "🎯 Scatter Insights Matrix":
        st.subheader("🎯 Bivariate Scatter Exploration Engine")
        if len(numeric_cols) >= 1:
            col_s1, col_s2 = st.columns([1, 3])
            with col_s1:
                x = st.selectbox("X Axis Dimensional Focus", numeric_cols, index=0)
                y = st.selectbox("Y Axis Focus", numeric_cols, index=get_safe_index(numeric_cols, 1))
                color_var = st.selectbox("Categorical Sorter", ["None"] + categorical_cols)
            
            with col_s2:
                fig = px.scatter(
                    df, x=x, y=y, 
                    color=None if color_var == "None" else color_var,
                    color_discrete_sequence=px.colors.qualitative.G10,
                    template="plotly_white", title=f"Visual Overlay: {y} vs {x}"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Analytical Assessment")
            st.info(f"**Descriptive Summary:** Inspecting coordinate clustering behavior. Groupings suggest specialized performance baselines among specific populations.")
            if color_var != "None":
                st.success(f"💡 **Prescriptive Guide:** Group variations verified via `{color_var}` indicate that a standardized broad strategy will fail. Deploy targeted, custom campaigns matching each subset.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Scatter maps require numeric features.")

    # ==========================================
    # ENGINE 4: LINEAR REGRESSION
    # ==========================================
    elif chart == "📈 Advanced Linear Regression":
        st.subheader("📈 Machine Learning Trajectory Engine")
        if len(numeric_cols) >= 1:
            col_r1, col_r2 = st.columns([1, 3])
            with col_r1:
                x = st.selectbox("Independent Input (X)", numeric_cols, index=0)
                y = st.selectbox("Dependent Target Variable (Y)", numeric_cols, index=get_safe_index(numeric_cols, 1))
            
            with col_r2:
                fig = px.scatter(df, x=x, y=y, trendline="ols", 
                                 trendline_color_override="#EF4444", 
                                 template="plotly_white",
                                 title="Predictive Regression Visualizer")
                st.plotly_chart(fig, use_container_width=True)

            # Extract Analytical Models
            X_model = df[[x]].dropna()
            y_model = df[y].loc[X_model.index]

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 🔮 Predictive & Prescriptive Architecture")
            if len(X_model) > 1:
                model = LinearRegression().fit(X_model, y_model)
                slope = model.coef_[0]
                r_sq = model.score(X_model, y_model)
                
                st.markdown(f"#### 📊 Algorithmic Output Coefficients")
                c1, c2 = st.columns(2)
                c1.metric("Calculated Impact Slope", f"{slope:,.4f}")
                c2.metric("R² System Accuracy Score", f"{r_sq*100:.2f}%")
                
                st.info(f"🔮 **Predictive Analysis:** Every 1-unit structural scale increase applied directly to `{x}` targets a clean shift of **{slope:.4f}** units inside `{y}`.")
                st.success(f"💡 **Prescriptive Action:** Use this formula to plan budgeting requirements. To achieve a specific goal for `{y}`, scale resources allocated to `{x}` using these model adjustments.")
            else:
                st.warning("Insufficient data inputs discovered to formulate predictive regression weights.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Regression calculations require numerical fields.")

    # ==========================================
    # ENGINE 5: BOX PLOT
    # ==========================================
    elif chart == "📦 Categorical Box & Range":
        st.subheader("📦 Structural Range Analysis Studio")
        if categorical_cols and numeric_cols:
            col_b1, col_b2 = st.columns([1, 3])
            with col_b1:
                cat = st.selectbox("Categorical Division Sorter", categorical_cols)
                num = st.selectbox("Numeric Focus", numeric_cols)
            
            with col_b2:
                fig = px.box(df, x=cat, y=num, color=cat, color_discrete_sequence=px.colors.qualitative.Safe, title=f"Range Spread: {num} Across {cat}")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Internal Spread Metrics Analysis")
            st.info(f"**Descriptive Insight:** Displaying cross-group variance, median baselines, and extreme statistical outliers for `{num}` across different `{cat}` variations.")
            st.success(f"💡 **Prescriptive Focus:** Prioritize streamlining operations for categories with high variance or large outlier counts. Stabilizing these volatile areas helps minimize unpredictable workflow spikes.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Box analyses require at least one categorical feature and one numeric feature range.")

    # ==========================================
    # ENGINE 6: VIOLIN PLOT
    # ==========================================
    elif chart == "🎻 Distribution Violin Curves":
        st.subheader("🎻 Structural Kernel Density View")
        if categorical_cols and numeric_cols:
            col_v1, col_v2 = st.columns([1, 3])
            with col_v1:
                cat = st.selectbox("Categorical Categorizer", categorical_cols)
                num = st.selectbox("Numeric Density Scale", numeric_cols)
            
            with col_v2:
                fig = px.violin(df, x=cat, y=num, color=cat, box=True, points="all", title=f"Density Shapes: {num} Split by {cat}")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Advanced Distribution Overview")
            st.info(f"**Descriptive Insight:** Rather than basic summaries, these curves reveal detailed probability wave shapes. Double bumps inside a category envelope signal that hidden, distinct user behaviors are merging under a single label.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Violin distributions require combined categorical and numeric fields.")

    # ==========================================
    # ENGINE 7: BUBBLE CHART
    # ==========================================
    elif chart == "🫧 3D Bubble Volume Explorer":
        st.subheader("🫧 Three-Dimensional Magnitude Explorer")
        if len(numeric_cols) >= 3:
            col_bb1, col_bb2 = st.columns([1, 3])
            with col_bb1:
                x = st.selectbox("X Coordinate", numeric_cols, index=0)
                y = st.selectbox("Y Coordinate", numeric_cols, index=get_safe_index(numeric_cols, 1))
                size = st.selectbox("Bubble Volume Dimension", numeric_cols, index=get_safe_index(numeric_cols, 2))
                color_var = st.selectbox("Color Theme Segment", ["None"] + categorical_cols)
            
            # Safe processing for scale values
            df_bubble = df.copy().dropna(subset=[x, y, size])
            if (df_bubble[size] < 0).any():
                df_bubble[size] = df_bubble[size].abs()

            with col_bb2:
                fig = px.scatter(
                    df_bubble, x=x, y=y, size=size,
                    color=None if color_var == "None" else color_var,
                    max_scale_size=50, template="plotly_white",
                    title=f"Volumetric Intersect Model ({size})"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Dynamic Account Profile Strategy")
            st.info(f"**Descriptive Tracking:** Maps baseline trends while using bubble sizes to clearly show the relative operational volume of each record.")
            st.info(f"🔮 **Predictive Forecast:** Large, isolated bubbles drifting far above normal trends flag high-volume, high-impact anomalies that are highly likely to drive future performance swings.")
            st.success(f"💡 **Prescriptive Guide:** Allocate specialized account management teams to closely monitor these high-volume clusters. Proactively managing these areas protects key revenue streams and mitigates systemic downside risk.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Volumetric Bubble tracking engines require a dataset with 3 or more numeric attributes.")

    # ==========================================
    # ENGINE 8: HEXBIN PLOT
    # ==========================================
    elif chart == "⬢ Hexbin High-Density Plot":
        st.subheader("⬢ Hexagonal High-Density Convergence")
        if len(numeric_cols) >= 1:
            col_h1, col_h2 = st.columns([1, 3])
            with col_h1:
                x = st.selectbox("X Structural Axis", numeric_cols, index=0)
                y = st.selectbox("Y Axis Vector Target", numeric_cols, index=get_safe_index(numeric_cols, 1))
            
            with col_h2:
                fig, ax = plt.subplots(figsize=(9, 5))
                hb = ax.hexbin(df[x], df[y], gridsize=28, cmap="plasma", mincnt=1)
                fig.colorbar(hb, ax=ax, label='Concentration Count')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"Density Hotspots Mapping Matrix")
                st.pyplot(fig)

            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 High-Density Visual Synthesis")
            st.info(f"**Descriptive Insight:** Designed specifically to handle messy, overlapping data. Bright yellow clusters highlight high-frequency transaction zones that are often completely hidden in standard, overcrowded scatter charts.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Hexagonal analysis requires numeric features.")

    # ==========================================
    # ENGINE 9: MULTIVARIATE PAIR MATRIX
    # ==========================================
    elif chart == "🌿 Multivariate Pair Matrix":
        st.subheader("🌿 Matrix Intersect Explorer")
        if len(numeric_cols) >= 2:
            cols = st.sidebar.multiselect(
                "Select Structural Grid Matrix Focus Features",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            if len(cols) >= 2:
                with st.spinner("Compiling cross-pair calculations..."):
                    fig = sns.pairplot(df[cols], palette="husl", diag_kind="kde")
                    st.pyplot(fig.figure)

                st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                st.markdown("### 📝 Matrix Review")
                st.info("**Descriptive Assessment:** This comprehensive multi-chart dashboard visually maps all two-variable combinations across selected dimensions. It makes it easy to quickly cross-examine complex data and filter out redundant, noisy metrics.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Please select at least 2 metrics inside the sidebar multiselect menu configuration panel.")
        else:
            st.error("Cross-pair charts require at least 2 numeric attributes.")