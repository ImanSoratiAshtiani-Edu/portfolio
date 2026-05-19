from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Loan Approval Classification",
    page_icon="🏦",
    layout="wide",
)


DATA_PATH = "data/raw/loan.csv"

TARGET_COL = "Loan_Status"
DROP_COLS = ["Loan_ID"]

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

COST_PRESETS: dict[str, dict[str, Any]] = {
    "Balanced": {
        "cost_fp": 1,
        "cost_fn": 1,
        "description": "FP and FN are treated equally.",
    },
    "Growth-focused Bank": {
        "cost_fp": 2,
        "cost_fn": 3,
        "description": "The bank also cares strongly about not losing good applicants.",
    },
    "Risk-aware Bank": {
        "cost_fp": 3,
        "cost_fn": 1,
        "description": "False approvals are considered more costly than false rejections.",
    },
    "Conservative Bank": {
        "cost_fp": 5,
        "cost_fn": 1,
        "description": "The bank strongly prefers avoiding risky approvals.",
    },
    "Very Conservative Bank": {
        "cost_fp": 10,
        "cost_fn": 1,
        "description": "The bank approves only when the model is highly confident.",
    },
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [col for col in df.columns if col not in DROP_COLS + [TARGET_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].map({"N": 0, "Y": 1})
    return X, y


def build_model_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    return model_pipeline


@st.cache_resource
def train_pipeline(path: str) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_data(path)
    X, y = build_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


def evaluate_threshold(
    y_true: pd.Series | np.ndarray,
    approval_proba: np.ndarray,
    threshold: float,
    cost_fp: int = 1,
    cost_fn: int = 1,
) -> dict[str, float | int]:
    y_pred = (approval_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 2),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_Y": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_Y": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_N": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_N": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "business_cost": int(cost_fp * fp + cost_fn * fn),
    }


def build_threshold_report(
    y_true: pd.Series | np.ndarray,
    approval_proba: np.ndarray,
    thresholds: np.ndarray,
    cost_fp: int,
    cost_fn: int,
) -> pd.DataFrame:
    rows = [
        evaluate_threshold(
            y_true=y_true,
            approval_proba=approval_proba,
            threshold=float(threshold),
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    approval_proba: np.ndarray,
    cost_fp: int,
    cost_fn: int,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.Series, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    report = build_threshold_report(
        y_true=y_true,
        approval_proba=approval_proba,
        thresholds=thresholds,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )

    best_row = report.sort_values(
        by=["business_cost", "accuracy", "FN", "FP"],
        ascending=[True, False, True, True],
    ).iloc[0]

    return float(best_row["threshold"]), best_row, report


def build_preset_comparison(
    y_true: pd.Series | np.ndarray,
    approval_proba: np.ndarray,
    presets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for preset_name, preset_info in presets.items():
        best_threshold, best_row, _ = find_best_threshold(
            y_true=y_true,
            approval_proba=approval_proba,
            cost_fp=preset_info["cost_fp"],
            cost_fn=preset_info["cost_fn"],
        )

        rows.append(
            {
                "preset": preset_name,
                "cost_fp": preset_info["cost_fp"],
                "cost_fn": preset_info["cost_fn"],
                "best_threshold": best_threshold,
                "business_cost": int(best_row["business_cost"]),
                "accuracy": best_row["accuracy"],
                "FP": int(best_row["FP"]),
                "FN": int(best_row["FN"]),
                "precision_Y": best_row["precision_Y"],
                "recall_Y": best_row["recall_Y"],
                "recall_N": best_row["recall_N"],
                "f1_N": best_row["f1_N"],
            }
        )

    return pd.DataFrame(rows)


def plot_confusion_matrix(metrics: dict[str, float | int]) -> plt.Figure:
    matrix = np.array(
        [
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted N", "Predicted Y"])
    ax.set_yticklabels(["Actual N", "Actual Y"])
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    return fig


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    approval_proba: np.ndarray,
    selected_metrics: dict[str, float | int],
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, approval_proba)
    auc = roc_auc_score(y_true, approval_proba)

    fp = selected_metrics["FP"]
    tn = selected_metrics["TN"]
    tp = selected_metrics["TP"]
    fn = selected_metrics["FN"]

    selected_fpr = fp / (fp + tn) if (fp + tn) else 0
    selected_tpr = tp / (tp + fn) if (tp + fn) else 0

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve - AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.scatter(
        selected_fpr,
        selected_tpr,
        s=90,
        label=f"Selected threshold = {selected_metrics['threshold']:.2f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall Approved_Y")
    ax.set_title("ROC Curve with Selected Threshold")
    ax.legend()
    ax.grid(True)

    return fig


df = load_data(DATA_PATH)
pipeline, X_train, X_test, y_train, y_test = train_pipeline(DATA_PATH)

approval_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, approval_proba)

st.title("🏦 Loan Approval Classification")
st.markdown(
    """
    This project predicts whether a loan application is likely to be approved.
    The page focuses not only on model accuracy, but also on the business trade-off
    between false approvals and false rejections.
    """
)

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

overview_col1.metric("Rows", f"{df.shape[0]}")
overview_col2.metric("Columns", f"{df.shape[1]}")
overview_col3.metric("ROC AUC", f"{roc_auc:.3f}")
overview_col4.metric("Model", "Logistic Regression")

with st.expander("Dataset preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Missing values"):
    missing_report = df.isna().sum().reset_index()
    missing_report.columns = ["column", "missing_count"]
    st.dataframe(missing_report, use_container_width=True)

st.subheader("Business Strategy Presets")

preset_comparison = build_preset_comparison(
    y_true=y_test,
    approval_proba=approval_proba,
    presets=COST_PRESETS,
)

st.dataframe(
    preset_comparison[
        [
            "preset",
            "cost_fp",
            "cost_fn",
            "best_threshold",
            "business_cost",
            "accuracy",
            "FP",
            "FN",
            "precision_Y",
            "recall_N",
        ]
    ],
    use_container_width=True,
)

selected_preset = st.selectbox(
    "Choose a business strategy",
    options=list(COST_PRESETS.keys()),
    index=2,
)

cost_fp = COST_PRESETS[selected_preset]["cost_fp"]
cost_fn = COST_PRESETS[selected_preset]["cost_fn"]

st.info(COST_PRESETS[selected_preset]["description"])
st.write(f"Cost formula: **{cost_fp} × FP + {cost_fn} × FN**")

best_threshold, best_row, threshold_report = find_best_threshold(
    y_true=y_test,
    approval_proba=approval_proba,
    cost_fp=cost_fp,
    cost_fn=cost_fn,
)

st.success(f"Recommended threshold for selected strategy: {best_threshold:.2f}")

threshold = st.slider(
    "Manual approval threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(best_threshold),
    step=0.01,
)

selected_metrics = evaluate_threshold(
    y_true=y_test,
    approval_proba=approval_proba,
    threshold=threshold,
    cost_fp=cost_fp,
    cost_fn=cost_fn,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
metric_col2.metric("Business Cost", f"{selected_metrics['business_cost']}")
metric_col3.metric("False Positive", f"{selected_metrics['FP']}")
metric_col4.metric("False Negative", f"{selected_metrics['FN']}")

metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)

metric_col5.metric("Precision Approved_Y", f"{selected_metrics['precision_Y']:.3f}")
metric_col6.metric("Recall Approved_Y", f"{selected_metrics['recall_Y']:.3f}")
metric_col7.metric("Recall Rejected_N", f"{selected_metrics['recall_N']:.3f}")
metric_col8.metric("F1 Rejected_N", f"{selected_metrics['f1_N']:.3f}")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.pyplot(plot_confusion_matrix(selected_metrics), use_container_width=True)

with chart_col2:
    st.pyplot(plot_roc_curve(y_test, approval_proba, selected_metrics), use_container_width=True)

with st.expander("Threshold report"):
    st.dataframe(
        threshold_report.sort_values(
            by=["business_cost", "accuracy", "FN", "FP"],
            ascending=[True, False, True, True],
        ).head(20),
        use_container_width=True,
    )

st.subheader("Interactive Loan Prediction")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
    married = st.selectbox("Married", sorted(df["Married"].dropna().unique()))
    dependents = st.selectbox("Dependents", sorted(df["Dependents"].dropna().unique()))
    education = st.selectbox("Education", sorted(df["Education"].dropna().unique()))

with input_col2:
    self_employed = st.selectbox("Self Employed", sorted(df["Self_Employed"].dropna().unique()))
    loan_amount_term = st.selectbox(
        "Loan Amount Term",
        sorted(df["Loan_Amount_Term"].dropna().unique()),
    )
    credit_history = st.selectbox(
        "Credit History",
        sorted(df["Credit_History"].dropna().unique()),
    )
    property_area = st.selectbox("Property Area", sorted(df["Property_Area"].dropna().unique()))

with input_col3:
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=500.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=500.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=150.0, step=10.0)

new_application = pd.DataFrame(
    [
        {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }
    ]
)

if st.button("Predict loan decision"):
    probability_approved = pipeline.predict_proba(new_application)[:, 1][0]
    predicted_label = int(probability_approved >= threshold)

    if predicted_label == 1:
        st.success(f"Prediction: Approved / Y — Probability: {probability_approved:.3f}")
    else:
        st.error(f"Prediction: Rejected / N — Probability: {probability_approved:.3f}")

    st.caption(
        "This prediction uses the currently selected threshold, not necessarily the default 0.50."
    )
