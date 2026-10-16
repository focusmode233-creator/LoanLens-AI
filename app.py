import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="HDFC Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# 2. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "loan_dataset.csv"
MODEL_PATH = BASE_DIR / "loan_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"


# ==================================================
# 3. LOAD DATA AND MODEL
# ==================================================

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    data["loan_status"] = (
        data["loan_status"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)

    return pd.DataFrame()


try:
    df = load_dataset()
    model = load_model()
    comparison = load_comparison()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.info(
        "Check that data/loan_dataset.csv and "
        "loan_model.pkl exist in your project folder."
    )
    st.stop()


# ==================================================
# 4. CUSTOM DESIGN
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #f5f7fa;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #aab4c3;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: #191f2b;
        border: 1px solid #303949;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab4c3;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🏦 Loan Intelligence")

st.sidebar.caption(
    "Machine Learning Project"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Analytics Dashboard",
        "🤖 Loan Prediction",
    ],
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Project Information**

    - Dataset: Public loan dataset
    - Records: 4,269
    - Algorithms: 2
    - Selected model: Random Forest
    """
)

st.sidebar.warning(
    "Educational prototype. Not an official "
    "HDFC Bank lending system."
)


# ==================================================
# 6. ANALYTICS DASHBOARD
# ==================================================

if page == "📊 Analytics Dashboard":

    st.markdown(
        '<div class="main-title">Loan Intelligence Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore loan application data and model performance.'
        '</div>',
        unsafe_allow_html=True,
    )

    total_applications = len(df)

    approved_count = int(
        (df["loan_status"] == "Approved").sum()
    )

    rejected_count = int(
        (df["loan_status"] == "Rejected").sum()
    )

    approval_rate = (
        approved_count / total_applications * 100
        if total_applications else 0
    )

    # ------------------------------
    # KPI CARDS
    # ------------------------------

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applications",
        f"{total_applications:,}",
    )

    col2.metric(
        "Approved",
        f"{approved_count:,}",
    )

    col3.metric(
        "Rejected",
        f"{rejected_count:,}",
    )

    col4.metric(
        "Historical Approval Rate",
        f"{approval_rate:.2f}%",
    )

    st.divider()

    # ------------------------------
    # APPLICATION DISTRIBUTION
    # ------------------------------

    st.markdown(
        '<div class="section-title">'
        'Application Status Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Approved vs Rejected")

        status_counts = (
            df["loan_status"]
            .value_counts()
            .reindex(
                ["Approved", "Rejected"],
                fill_value=0,
            )
        )

        st.bar_chart(
            status_counts,
            x_label="Loan Status",
            y_label="Number of Applications",
        )

    with chart_col2:

        st.subheader("Education Distribution")

        education_counts = (
            df["education"]
            .value_counts()
        )

        st.bar_chart(
            education_counts,
            x_label="Education",
            y_label="Number of Applicants",
        )

    st.divider()

    # ------------------------------
    # NUMERICAL ANALYSIS
    # ------------------------------

    st.markdown(
        '<div class="section-title">'
        'Applicant Financial Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    financial_col1, financial_col2 = st.columns(2)

    with financial_col1:

        st.subheader("Annual Income Distribution")

        st.bar_chart(
            df["income_annum"]
            .value_counts(bins=10)
            .sort_index(),
            x_label="Income Range",
            y_label="Number of Applicants",
        )

    with financial_col2:

        st.subheader("Loan Amount Distribution")

        st.bar_chart(
            df["loan_amount"]
            .value_counts(bins=10)
            .sort_index(),
            x_label="Loan Amount Range",
            y_label="Number of Applications",
        )

    st.divider()

    # ------------------------------
    # MODEL COMPARISON
    # ------------------------------

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Model Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    if not comparison.empty:

        st.write(
            "Accuracy of the algorithms on the held-out "
            "test dataset used during model evaluation."
        )

        st.bar_chart(
            comparison.set_index("Model")["Accuracy (%)"],
            y_label="Accuracy (%)",
        )

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

        st.success(
            "Random Forest achieved 97.78% accuracy "
            "in our current evaluation."
        )

    else:

        st.warning(
            "Model comparison report was not found."
        )

    st.caption(
        "Dataset statistics describe this public dataset, "
        "not actual HDFC Bank operations. Test accuracy "
        "does not guarantee performance on new applications."
    )


# ==================================================
# 7. LOAN PREDICTION PAGE
# ==================================================

elif page == "🤖 Loan Prediction":

    st.markdown(
        '<div class="main-title">Loan Approval Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter applicant information to obtain a '
        'machine learning prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This application uses a trained Random Forest "
        "model and a public educational dataset. "
        "It does not make actual bank lending decisions."
    )

    st.divider()

    st.subheader("Applicant Information")

    with st.form("loan_application_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Personal Information")

            no_of_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
            )

            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"],
            )

            self_employed = st.selectbox(
                "Self Employed",
                ["Yes", "No"],
            )

            st.markdown("#### Loan Information")

            income_annum = st.number_input(
                "Annual Income",
                min_value=0,
                value=500000,
                step=50000,
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=1000000,
                step=50000,
            )

            loan_term = st.number_input(
                "Loan Term",
                min_value=1,
                value=10,
                step=1,
            )

        with col2:

            st.markdown("#### Credit Information")

            cibil_score = st.number_input(
                "CIBIL Score",
                min_value=300,
                max_value=900,
                value=700,
                step=1,
            )

            st.markdown("#### Asset Information")

            residential_assets_value = st.number_input(
                "Residential Assets Value",
                min_value=0,
                value=1000000,
                step=100000,
            )

            commercial_assets_value = st.number_input(
                "Commercial Assets Value",
                min_value=0,
                value=500000,
                step=100000,
            )

            luxury_assets_value = st.number_input(
                "Luxury Assets Value",
                min_value=0,
                value=500000,
                step=100000,
            )

            bank_asset_value = st.number_input(
                "Bank Asset Value",
                min_value=0,
                value=300000,
                step=50000,
            )

        submitted = st.form_submit_button(
            "Predict Loan Status",
            type="primary",
            use_container_width=True,
        )

    # ------------------------------
    # MAKE PREDICTION
    # ------------------------------

    if submitted:

        applicant = pd.DataFrame(
            [
                {
                    "no_of_dependents": no_of_dependents,
                    "education": education,
                    "self_employed": self_employed,
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value":
                        residential_assets_value,
                    "commercial_assets_value":
                        commercial_assets_value,
                    "luxury_assets_value":
                        luxury_assets_value,
                    "bank_asset_value": bank_asset_value,
                }
            ]
        )

        try:

            prediction = model.predict(applicant)[0]

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'Prediction Results'
                '</div>',
                unsafe_allow_html=True,
            )

            if prediction == "Approved":

                st.success(
                    "✅ Model Prediction: APPROVED"
                )

            else:

                st.error(
                    "❌ Model Prediction: REJECTED"
                )

            # Show model probabilities when available
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    applicant
                )[0]

                probability_df = pd.DataFrame(
                    {
                        "Outcome": model.classes_,
                        "Model Probability (%)":
                            probabilities * 100,
                    }
                )

                probability_df[
                    "Model Probability (%)"
                ] = probability_df[
                    "Model Probability (%)"
                ].round(2)

                st.subheader(
                    "Prediction Probabilities"
                )

                st.bar_chart(
                    probability_df.set_index("Outcome"),
                    y_label="Probability (%)",
                )

                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True,
                )

            st.caption(
                "The prediction is based on patterns in "
                "the training dataset. Model probabilities "
                "are not guarantees of actual loan approval."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )