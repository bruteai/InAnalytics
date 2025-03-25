import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import pickle

# -------------------------------
# PAGE CONFIG & CUSTOM STYLING
# -------------------------------
st.set_page_config(
    page_title="InAnalytics: Full Dashboard",
    page_icon="✅",
    layout="wide"
)

custom_css = """
<style>
/* Hide Streamlit's default menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Light blue gradient background for entire page & sidebar */
[data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
    background: linear-gradient(135deg, #c2e9fb, #e0f7fa) no-repeat center center fixed;
    background-size: cover;
    color: #000000 !important; /* Force black text for readability */
    border: none !important;
    box-shadow: none !important;
}

/* Top bar styling with a semi-transparent overlay so the gradient shows through */
.top-bar {
    background-color: rgba(194, 233, 251, 0.9);
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-bar a {
    margin-right: 1rem;
    text-decoration: none;
    color: #0072C6;
    font-weight: 500;
}
.top-bar a:hover {
    text-decoration: underline;
}

/* Override main container text color - black */
section.main > div {
    color: #000000;
}

/* Metric card styling - white cards with black text */
.metric-card {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    text-align: center;
    margin-bottom: 1rem;
    color: #000;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: bold;
    color: #2E86C1;
}
.metric-label {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.25rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------
# TOP NAV BAR
# -------------------------------
st.markdown(
    """
    <div class="top-bar">
        <div>
            <strong style="margin-right: 2rem;">InAnalytics</strong>
        </div>
        <div>
            <a href="#">Schedule Email Report</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR MENU
# (No "CLV Analysis" tab, as requested)
# -------------------------------
st.sidebar.title("InAnalytics Dashboard")

menu_options = [
    "Home",
    "ML Models",
    "CLV Prediction",
    "Segmentation",
    "Data Analysis",
    "Reports",
    "Settings"
]
selected_page = st.sidebar.radio("", menu_options)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Filters")
date_range = st.sidebar.date_input("Select Date Range")

st.sidebar.markdown("---")
st.sidebar.markdown("### Book a Demo")
if st.sidebar.button("Book Live Demo"):
    st.session_state["show_demo_modal"] = True

# Initialize modal session state
if "show_demo_modal" not in st.session_state:
    st.session_state["show_demo_modal"] = False

# ----------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------
if selected_page == "Home":
    st.header("Home Overview")

    # Top row of key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">$65,237</div>
                <div class="metric-label">Total Revenue (MTD)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">1,492</div>
                <div class="metric-label">Active Policies</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">$2,310</div>
                <div class="metric-label">Avg. CLV</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">23%</div>
                <div class="metric-label">Predicted Churn Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Monthly Revenue Trends")

    # Generate random monthly data for demonstration
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    df_revenue = pd.DataFrame({
        "month": months,
        "revenue": np.random.randint(30000, 70000, size=12)
    })
    chart = (
        alt.Chart(df_revenue)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month"),  # No forced labelColor
            y=alt.Y("revenue:Q", title="Revenue ($)"),
            tooltip=["month:T", "revenue:Q"]
        )
        .properties(width="container", height=300)
    )
    chart = chart.configure_axis(
        labelColor="#000",  # Axis labels in black
        titleColor="#000",
        gridColor="#AAA"
    ).configure_view(strokeOpacity=0)

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Data Preview")
    sample_data = pd.DataFrame({
        "Policy_ID": [f"POL{i:04d}" for i in range(1, 11)],
        "Customer_Age": np.random.randint(20, 80, 10),
        "Policy_Premium": np.round(np.random.uniform(300, 2000, 10), 2),
        "Predicted_CLV": np.round(np.random.uniform(1000, 5000, 10), 2)
    })
    st.dataframe(sample_data)

# ----------------------------------------------------
# PAGE 2: ML MODELS
# ----------------------------------------------------
elif selected_page == "ML Models":
    st.header("Machine Learning Models (Performance)")

    METRICS_PATH = os.path.join("..", "models", "model_metrics.csv")
    if not os.path.exists(METRICS_PATH):
        st.error("No model_metrics.csv found. Please run model_training.py first.")
    else:
        metrics_df = pd.read_csv(METRICS_PATH)

        st.markdown("### Model Performance Comparison")
        st.dataframe(metrics_df)

        st.write("#### RMSE Comparison")
        rmse_chart = (
            alt.Chart(metrics_df)
            .mark_bar(color="#4e79a7")
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("RMSE:Q", title="RMSE"),
                tooltip=["model", "RMSE", "R2"]
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(rmse_chart, use_container_width=False)

        st.write("#### R² Comparison")
        r2_chart = (
            alt.Chart(metrics_df)
            .mark_bar(color="#f28e2c")
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("R2:Q", title="R²"),
                tooltip=["model", "R2"]
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(r2_chart, use_container_width=False)

# ----------------------------------------------------
# PAGE 3: CLV PREDICTION
# ----------------------------------------------------
elif selected_page == "CLV Prediction":
    st.header("Customer Lifetime Value Prediction")

    model_options = {
        "Linear Regression": "model_linearregression.pkl",
        "Random Forest": "model_randomforest.pkl",
        "Gradient Boosting": "model_gradientboosting.pkl"
    }

    selected_model_name = st.selectbox(
        "Choose a model for prediction:",
        list(model_options.keys())
    )

    model_path = os.path.join("..", "models", model_options[selected_model_name])
    if not os.path.exists(model_path):
        st.warning(f"Selected model file not found: {model_path}")
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        st.write("#### Enter Key Customer Features")

        colA, colB, colC = st.columns(3)
        with colA:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            num_claims = st.slider("Number of Claims", 0, 5, 1, step=1)

        with colB:
            policy_premium = st.slider("Policy Premium ($)", 300.0, 2000.0, 1000.0, step=50.0)
            frequency = st.slider("Frequency (transactions)", 1, 12, 3, step=1)

        with colC:
            recency = st.slider("Recency (days)", 1, 365, 100, step=1)
            monetary = st.slider("Monetary (total spend)", 500.0, 20000.0, 5000.0, step=500.0)

        gender = st.radio("Gender", ["Male", "Female"], index=0)
        province_list = [
            "Alberta", "British Columbia", "Manitoba", "New Brunswick",
            "Newfoundland and Labrador", "Nova Scotia", "Ontario",
            "Prince Edward Island", "Quebec", "Saskatchewan"
        ]
        province = st.selectbox("Province", province_list, index=0)
        feedback_list = [
            "Excellent service", "Very satisfied", "Satisfied",
            "Neutral", "Unsatisfied", "Poor service"
        ]
        feedback = st.selectbox("Customer Feedback", feedback_list, index=0)

        # Construct Feature Vector
        all_columns = [
            "Age", "Policy_Premium", "Num_Claims", "Frequency", "Recency", "Monetary",
            "Gender_Female", "Gender_Male",
            "Province_Alberta", "Province_British Columbia", "Province_Manitoba",
            "Province_New Brunswick", "Province_Newfoundland and Labrador",
            "Province_Nova Scotia", "Province_Ontario", "Province_Prince Edward Island",
            "Province_Quebec", "Province_Saskatchewan",
            "Customer_Feedback_Excellent service", "Customer_Feedback_Neutral",
            "Customer_Feedback_Poor service", "Customer_Feedback_Satisfied",
            "Customer_Feedback_Unsatisfied", "Customer_Feedback_Very satisfied"
        ]

        def create_feature_vector():
            row_data = {col: 0 for col in all_columns}
            row_data["Age"] = age
            row_data["Policy_Premium"] = policy_premium
            row_data["Num_Claims"] = num_claims
            row_data["Frequency"] = frequency
            row_data["Recency"] = recency
            row_data["Monetary"] = monetary

            # Gender
            if gender == "Male":
                row_data["Gender_Male"] = 1
            else:
                row_data["Gender_Female"] = 1

            # Province
            province_col = f"Province_{province}"
            row_data[province_col] = 1

            # Feedback
            feedback_col = f"Customer_Feedback_{feedback}"
            row_data[feedback_col] = 1

            return pd.DataFrame([row_data])

        if st.button("Predict CLV"):
            input_df = create_feature_vector()
            prediction = model.predict(input_df)[0]
            st.success(f"**Predicted CLV:** ${prediction:,.2f}")

# ----------------------------------------------------
# PAGE 4: SEGMENTATION
# ----------------------------------------------------
elif selected_page == "Segmentation":
    st.header("Customer Segmentation Overview")

    SEGMENTS_PATH = os.path.join("..", "data", "customer_segments.csv")
    if not os.path.exists(SEGMENTS_PATH):
        st.error("No customer_segments.csv found. Please run customer_segmentation.py first.")
    else:
        df_segments = pd.read_csv(SEGMENTS_PATH)

        df_segments["Segment"] = df_segments["Segment"].astype(str)
        df_segments["Segment"] = "Segment " + df_segments["Segment"]

        st.subheader("Segment Counts")
        segment_counts = df_segments["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        st.dataframe(segment_counts)

        st.write("### Distribution of Segments")
        base_chart = alt.Chart(segment_counts).encode(
            x=alt.X("Segment:N", title="Segment"),
            y=alt.Y("Count:Q", title="Number of Customers")
        )
        bar_layer = base_chart.mark_bar(color="#2E86C1").encode(
            tooltip=["Segment:N", "Count:Q"]
        )
        text_layer = base_chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-2,
            color="white"
        ).encode(
            text="Count:Q"
        )
        final_chart = (bar_layer + text_layer).properties(width="container", height=300)
        final_chart = final_chart.configure_axis(
            labelColor="#000",   # black axis labels
            titleColor="#000",
            labelFontSize=12,
            titleFontSize=14,
            gridColor="#AAA"
        ).configure_view(strokeOpacity=0)
        st.altair_chart(final_chart, use_container_width=True)

        st.write("### Average RFM by Segment")
        if all(col in df_segments.columns for col in ["Recency", "Frequency", "Monetary"]):
            rfm_stats = df_segments.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
            st.dataframe(rfm_stats.style.format({"Recency": "{:.1f}", "Frequency": "{:.1f}", "Monetary": "{:.1f}"}))

            st.write("### Monetary vs. Frequency by Segment")
            scatter_chart = (
                alt.Chart(df_segments)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("Frequency:Q"),
                    y=alt.Y("Monetary:Q"),
                    color="Segment:N",
                    tooltip=["Segment", "Frequency", "Monetary", "Recency"]
                )
                .properties(width="container", height=400)
            )
            scatter_chart = scatter_chart.configure_axis(
                labelColor="#000",
                titleColor="#000",
                gridColor="#AAA"
            ).configure_view(strokeOpacity=0)
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.warning("RFM columns (Recency, Frequency, Monetary) not found in the segments file.")

# ----------------------------------------------------
# PAGE 5: DATA ANALYSIS
# ----------------------------------------------------
elif selected_page == "Data Analysis":
    st.header("Insurance Data Analysis")

    DATA_ANALYSIS_PATH = os.path.join("..", "data", "insurance_analysis.csv")
    if not os.path.exists(DATA_ANALYSIS_PATH):
        st.error("No insurance_analysis.csv found. Please place the file in ../data/.")
    else:
        df_analysis = pd.read_csv(DATA_ANALYSIS_PATH)

        if "Transaction_Date" in df_analysis.columns:
            df_analysis["Transaction_Date"] = pd.to_datetime(df_analysis["Transaction_Date"], errors="coerce")

        st.markdown("### 1. Policy Type Distribution (Pie Chart)")
        if "Policy_Type" in df_analysis.columns:
            policy_counts = df_analysis["Policy_Type"].value_counts().reset_index()
            policy_counts.columns = ["Policy_Type", "Count"]
            pie_chart = (
                alt.Chart(policy_counts)
                .mark_arc(innerRadius=50)
                .encode(
                    theta="Count:Q",
                    color="Policy_Type:N",
                    tooltip=["Policy_Type:N", "Count:Q"]
                )
                .properties(width=400, height=400)
            )
            st.altair_chart(pie_chart, use_container_width=False)
        else:
            st.warning("No 'Policy_Type' column found.")

        st.markdown("### 2. Average Claim Amount by Region (Bar Chart)")
        if all(col in df_analysis.columns for col in ["Region", "Claim_Amount"]):
            avg_claims = df_analysis.groupby("Region")["Claim_Amount"].mean().reset_index()
            avg_claims.columns = ["Region", "Avg_Claim_Amount"]

            bar_chart = (
                alt.Chart(avg_claims)
                .mark_bar(color="#2E86C1")
                .encode(
                    x=alt.X("Region:N", sort="-y", title="Region"),
                    y=alt.Y("Avg_Claim_Amount:Q", title="Average Claim Amount ($)"),
                    tooltip=["Region:N", "Avg_Claim_Amount:Q"]
                )
                .properties(width="container", height=300)
            )
            bar_chart = bar_chart.configure_axis(
                labelColor="#000",
                titleColor="#000",
                labelFontSize=12,
                titleFontSize=14,
                gridColor="#AAA"
            ).configure_view(strokeOpacity=0)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.warning("No 'Region' or 'Claim_Amount' columns found.")

        st.markdown("### 3. Claims Over Time (Line Chart)")
        if all(col in df_analysis.columns for col in ["Transaction_Date", "Claim_Amount"]):
            df_analysis["Month"] = df_analysis["Transaction_Date"].dt.to_period("M")
            monthly_claims = df_analysis.groupby("Month")["Claim_Amount"].sum().reset_index()
            monthly_claims["Month"] = monthly_claims["Month"].astype(str)

            line_chart = (
                alt.Chart(monthly_claims)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Month:N", title="Month", sort=None),
                    y=alt.Y("Claim_Amount:Q", title="Total Claims ($)"),
                    tooltip=["Month:N", "Claim_Amount:Q"]
                )
                .properties(width="container", height=300)
            )
            line_chart = line_chart.configure_axis(
                labelColor="#000",
                titleColor="#000",
                gridColor="#AAA"
            ).configure_view(strokeOpacity=0)
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.warning("No 'Transaction_Date' or 'Claim_Amount' columns found.")

        st.markdown("### 4. Anomaly Detection (High Claim Amounts)")
        if "Claim_Amount" in df_analysis.columns:
            threshold = df_analysis["Claim_Amount"].quantile(0.95)
            anomalies = df_analysis[df_analysis["Claim_Amount"] > threshold]

            st.write(f"Claims above the 95th percentile (Threshold = ${threshold:,.2f}):")
            st.dataframe(anomalies.head(20))
        else:
            st.warning("No 'Claim_Amount' column for anomaly detection.")

# ----------------------------------------------------
# PAGE 6: REPORTS
# ----------------------------------------------------
elif selected_page == "Reports":
    st.header("Reports")
    st.write("Generate or schedule custom reports here.")

# ----------------------------------------------------
# PAGE 7: SETTINGS
# ----------------------------------------------------
elif selected_page == "Settings":
    st.header("Settings")
    st.write("Adjust user settings, billing, or integrations here.")

# -------------------------------
# BOOK DEMO MODAL
# -------------------------------
if st.session_state["show_demo_modal"]:
    with st.modal("Book a Live Demo", key="demo_modal"):
        st.write("Get a personalized walkthrough of our full InAnalytics platform.")
        st.write("We'll show you how to optimize your insurance strategy with advanced analytics.")
        if st.button("Close"):
            st.session_state["show_demo_modal"] = False

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("© 2025 InAnalytics. All rights reserved.")
