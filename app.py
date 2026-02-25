import streamlit as st
import pandas as pd
import pickle
import tempfile
import os

import Orange
from Orange.data import Table

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Change this to your freshly re-saved model filename
MODEL_PATH = "gradient_boosting_fresh.pkcls"


@st.cache_resource
def load_orange_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_user_input() -> pd.DataFrame:
    st.subheader("Enter patient details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
    smoking_history = st.selectbox(
        "Smoking history",
        ["Never", "No info", "Former", "current"]
    )

    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=24.5, step=0.1)
    hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=20.0, value=5.8, step=0.1)
    blood_glucose = st.number_input("Blood glucose level", min_value=20.0, max_value=600.0, value=110.0, step=1.0)

    # EXACT column names from your dataset (case-sensitive)
    df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Hypertension": hypertension,
        "Heart_disease": heart_disease,
        "Smoking_history": smoking_history,
        "BMI": bmi,
        "HbA1c_level": hba1c,
        "Blood_glucose_level": blood_glucose
    }])
    return df


def predict_with_orange(model, input_df: pd.DataFrame):
    """
    Convert one-row pandas DataFrame to Orange Table via temporary CSV, then predict.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
        temp_path = tmp.name
        input_df.to_csv(tmp, index=False)

    try:
        data = Table.from_file(temp_path)

        # Prediction + probabilities if supported
        try:
            pred_vals, pred_probs = model(data, ret=Orange.base.Model.ValueProbs)
            pred_label = str(pred_vals[0])

            class_labels = []
            if hasattr(model, "domain") and model.domain.class_var is not None:
                cv = model.domain.class_var
                if hasattr(cv, "values") and cv.values:
                    class_labels = list(cv.values)

            return pred_label, pred_probs[0], class_labels

        except Exception:
            pred_vals = model(data)
            pred_label = str(pred_vals[0])
            return pred_label, None, []

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- UI ----------------

st.title("Diabetes Risk Predictor")
st.write("Enter a single patient's values and click **Predict**.")

# Load model
try:
    model = load_orange_model(MODEL_PATH)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Could not load model file '{MODEL_PATH}': {e}")
    st.stop()

# Debug section (optional but very useful)
with st.expander("Debug: Model expected domain"):
    try:
        if hasattr(model, "domain"):
            st.write("### Attributes")
            for a in model.domain.attributes:
                if hasattr(a, "values") and a.values:
                    st.write(f"- {a.name} (discrete): {list(a.values)}")
                else:
                    st.write(f"- {a.name} (continuous)")
            if model.domain.class_var is not None:
                cv = model.domain.class_var
                if hasattr(cv, "values") and cv.values:
                    st.write(f"### Class variable: {cv.name} -> {list(cv.values)}")
                else:
                    st.write(f"### Class variable: {cv.name}")
    except Exception as e:
        st.warning(f"Could not inspect model domain: {e}")

with st.form("prediction_form"):
    input_df = get_user_input()
    submitted = st.form_submit_button("Predict")

if submitted:
    st.markdown("### Input values")
    st.dataframe(input_df, use_container_width=True)

    try:
        pred_label, probs, class_labels = predict_with_orange(model, input_df)

        st.markdown("## Prediction Result")
        if pred_label.strip().lower() == "yes":
            st.error(f"Predicted class: **{pred_label}**")
        else:
            st.success(f"Predicted class: **{pred_label}**")

        if probs is not None:
            probs = [float(p) for p in probs]
            st.markdown("### Class probabilities")

            if class_labels and len(class_labels) == len(probs):
                prob_df = pd.DataFrame({"Class": class_labels, "Probability": probs})
                st.dataframe(prob_df, use_container_width=True)

                prob_map = dict(zip(class_labels, probs))
                if "Yes" in prob_map:
                    st.info(f"Probability (Yes): **{prob_map['Yes']:.3f}**")
                if "No" in prob_map:
                    st.info(f"Probability (No): **{prob_map['No']:.3f}**")
            else:
                prob_df = pd.DataFrame({
                    "Class": [f"Class_{i}" for i in range(len(probs))],
                    "Probability": probs
                })
                st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Possible cause: model expects preprocessed domain instead of raw columns.")
        st.exception(e)

st.caption("Labels and column names must exactly match the training schema.")