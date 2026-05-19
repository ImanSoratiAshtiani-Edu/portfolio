import streamlit as st

st.set_page_config(
    page_title="Applied Machine Learning Portfolio",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Applied Machine Learning Portfolio")

st.markdown(
    """
    A collection of practical machine learning projects focused on model development,
    evaluation, business-aware decision making, and interactive deployment.

    This portfolio currently includes supervised learning projects built with
    Python, scikit-learn, and Streamlit.
    """
)

st.subheader("Project Categories")

st.markdown(
    """
    ### Supervised Machine Learning

    Projects in this section focus on classification and regression problems,
    including preprocessing pipelines, model evaluation, threshold tuning,
    and business interpretation.
    """
)

st.subheader("Available Projects")

st.markdown(
    """
    #### 🏦 Project 1 — Loan Approval Classification

    A supervised classification project for predicting loan approval decisions.

    Key topics:
    - preprocessing pipeline
    - logistic regression
    - ROC AUC analysis
    - business-cost threshold tuning
    - dynamic confusion matrix
    - interactive prediction form
    """
)
