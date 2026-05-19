import streamlit as st

st.set_page_config(
    page_title="Applied Machine Learning Portfolio",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Applied Machine Learning Portfolio")

st.markdown(
    """
    A structured portfolio of practical machine learning projects focused on
    model development, evaluation, business-aware decision making, and interactive deployment.
    """
)

st.divider()

st.subheader("Project Categories")

cat_col1, cat_col2, cat_col3 = st.columns(3)

with cat_col1:
    st.markdown(
        """
        ### 🎯 Supervised Learning
        Classification and regression projects with preprocessing pipelines,
        model evaluation, and business interpretation.
        """
    )

with cat_col2:
    st.markdown(
        """
        ### 📊 Model Evaluation
        Threshold tuning, ROC analysis, confusion matrices, error analysis,
        and metric selection.
        """
    )

with cat_col3:
    st.markdown(
        """
        ### 🚀 Deployment
        Interactive Streamlit applications designed for portfolio presentation
        and practical decision support.
        """
    )

st.divider()

st.subheader("Available Projects")

project_col1, project_col2 = st.columns([1, 1])

with project_col1:
    st.markdown(
        """
        ### 🏦 Project 1 — Loan Approval Classification

        **Type:** Binary Classification  
        **Model:** Logistic Regression  
        **Frameworks:** scikit-learn, pandas, Streamlit  

        This project predicts loan approval decisions and explores the trade-off
        between false approvals and false rejections using business-cost based
        threshold tuning.
        """
    )

    st.page_link(
        "pages/01_Loan_Approval.py",
        label="Open Loan Approval Project",
        icon="🏦",
    )

with project_col2:
    st.markdown(
        """
        ### 🧩 Next Projects

        Planned additions:

        - Customer churn prediction
        - House price regression
        - Medical risk classification
        - Text classification
        - Model comparison dashboard
        """
    )

st.divider()

st.caption(
    "Portfolio repository: ImanSoratiAshtiani-Edu/portfolio"
)
