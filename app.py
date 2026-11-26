# ==================================================
# Heart Attack Prediction App (Streamlit)
# ==================================================
# ---------- 1. Import libraries ----------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
# Files are read from the same folder as app.py
MODEL_PATH = Path(__file__).resolve().with_name("heart_attack_model.pkl")
# ---------- 2. Page settings (must be the first Streamlit command) ----------
st.set_page_config(page_title="Heart Attack Prediction", page_icon="❤️", layout="wide")
# ---------- 3. Medical colour theme (simple CSS) ----------
st.markdown("""
<style>
    .stApp { background-color: #F4F8F6; color: #173E35; }
    h1, h2, h3 { color: #173E35; }
    .main-header { background: linear-gradient(120deg, #123F36, #237C67); padding: 25px;
        border-radius: 12px; text-align: center; margin-bottom: 20px; }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: #E2F0EB; margin: 5px 0 0 0; font-size: 17px; }
    .stButton > button { background-color: #176B55; color: white;
        font-size: 18px; border-radius: 8px; width: 100%; }
    .stButton > button:hover { background-color: #123F36; color: white; }
</style>
""", unsafe_allow_html=True)
# ---------- 4. Load the saved model (pipeline: preprocessing + tuned Logistic Regression) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
model = load_model()
def predict_heart_attack(model, input_data):
    """Turn the answers into a one-row DataFrame and return (predicted class, probability of 'Yes')."""
    input_df = pd.DataFrame(input_data).reindex(columns=model.feature_names_in_)          # input_data holds one-item lists, e.g. {'Sex': ['Female'], ...}
    result = model.predict(input_df)[0]
    yes_index = list(model.classes_).index('Yes')
    probability = float(model.predict_proba(input_df)[0][yes_index])
    return result, probability
# ---------- 5. Dropdown options: copied exactly from the dataset ----------
# The model's OneHotEncoder only accepts values it saw during training.
choices = {
"Sex": ["Female", "Male"],
"AgeCategory": ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
"Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
"Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
"Age 80 or older"],
"GeneralHealth": ["Excellent", "Very good", "Good", "Fair", "Poor"],
"LastCheckupTime": ["Within past year (anytime less than 12 months ago)",
"Within past 2 years (1 year but less than 2 years ago)",
"Within past 5 years (2 years but less than 5 years ago)",
"5 or more years ago"],
"PhysicalActivities": ["Yes", "No"],
"HadAngina": ["No", "Yes"],
"HadAsthma": ["No", "Yes"],
"HadCOPD": ["No", "Yes"],
"HadDepressiveDisorder": ["No", "Yes"],
"ChestScan": ["No", "Yes"],
"SmokerStatus": ["Never smoked", "Former smoker",
"Current smoker - now smokes some days",
"Current smoker - now smokes every day"],
"ECigaretteUsage": ["Never used e-cigarettes in my entire life", "Not at all (right now)",
"Use them some days", "Use them every day"],
"AlcoholDrinkers": ["No", "Yes"],
"HIVTesting": ["No", "Yes"],
"TetanusLast10Tdap": ["Yes, received Tdap",
"Yes, received tetanus shot, but not Tdap",
"Yes, received tetanus shot but not sure what type",
"No, did not receive any tetanus shot in the past 10 years"],
"HighRiskLastYear": ["No", "Yes"],
"CovidPos": ["No", "Yes", "Tested positive using home test without a health professional"],
}
# ---------- 6. Explanations shown in the "?" tooltips ----------
help_text = {
'GeneralHealth': 'How the person would rate their own overall health.',
'LastCheckupTime': 'When the person last visited a doctor for a routine check-up.',
'BMI': 'BMI = weight (kg) / height (m) squared. Below 18.5 = underweight, '
'18.5-24.9 = normal, 25-29.9 = overweight, 30 or more = obese.',
'HadAngina': 'Angina is chest pain or pressure caused by reduced blood flow to the heart. '
'It is a warning sign of heart disease.',
'HadAsthma': 'Asthma is a long-term condition where the airways narrow and swell, '
'making breathing difficult.',
'HadCOPD': 'COPD (Chronic Obstructive Pulmonary Disease) is a long-term lung disease, '
'such as emphysema or chronic bronchitis, often caused by smoking.',
'HadDepressiveDisorder': 'Ever been told by a health professional that they have depression.',
'ChestScan': 'A CT (Computed Tomography) scan is a detailed X-ray image of the chest.',
'PhysicalActivities': 'Any exercise or physical activity outside of regular work, '
'such as walking, running or sports.',
'SleepHours': 'Adults are generally advised to sleep 7-9 hours per night.',
'ECigaretteUsage': "'Not at all (right now)' means the person used them before but not now.",
'HIVTesting': 'Whether the person has ever had an HIV test (not the result).',
'TetanusLast10Tdap': 'Tdap is a combined vaccine that protects against tetanus, '
'diphtheria and whooping cough.',
'HighRiskLastYear': 'Survey question about behaviours linked to a higher risk of HIV '
'in the past 12 months.',
}
# ---------- 7. Input groups: (section title, [column 1 fields, column 2 fields, ...]) ----------
input_groups = [
("👤 Demographics", [
[("Sex", "Sex")],
[("AgeCategory", "Age Category")]]),
("🩺 General Health", [
[("GeneralHealth", "General Health")],
[("LastCheckupTime", "Last Routine Check-up")],
[("BMI", "BMI (Body Mass Index)")]]),
("📋 Medical History", [
[("HadAngina", "Ever had Angina?"), ("HadAsthma", "Ever had Asthma?")],
[("HadCOPD", "Ever had COPD?"), ("HadDepressiveDisorder", "Ever had a Depressive Disorder?")],
[("ChestScan", "Ever had a Chest CT Scan?")]]),
("🏃 Lifestyle", [
[("PhysicalActivities", "Physical Activity in Past 30 Days?"),
("SleepHours", "Average Sleep Hours (per day)")],
[("SmokerStatus", "Smoking Status"), ("AlcoholDrinkers", "Drank Alcohol in Past 30 Days?")],
[("ECigaretteUsage", "E-Cigarette Use")]]),
("💉 Other Health Indicators", [
[("HIVTesting", "Ever Been Tested for HIV?"),
("TetanusLast10Tdap", "Tetanus Vaccine in Past 10 Years")],
[("HighRiskLastYear", "High-Risk Situation in Past Year?"),
("CovidPos", "Ever Tested Positive for COVID-19?")]]),
]
# Default answer for each dropdown (any field not listed starts at its first option)
defaults = {
"Sex": "Female", "AgeCategory": "Age 18 to 24", "GeneralHealth": "Excellent",
"LastCheckupTime": "Within past year (anytime less than 12 months ago)",
"PhysicalActivities": "Yes", "SmokerStatus": "Never smoked",
"ECigaretteUsage": "Never used e-cigarettes in my entire life",
"TetanusLast10Tdap": "Yes, received Tdap",
}
# Flat list of (field, label) in the order they appear on the page
FIELDS = [(field, label) for _, columns in input_groups for column in columns for field, label in column]
def main():
    # ---------- 8. Header and disclaimer ----------
    st.markdown("""
    <div class="main-header">
    <h1>❤️ Heart Attack Prediction App</h1>
    <p>An educational machine learning project using a tuned Logistic Regression model</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning(
        "⚠️ **Disclaimer:** This app is an educational machine learning project and is "
        "**not a medical diagnostic tool**. The model was trained on self-reported survey data "
        "(CDC BRFSS 2022). The target shows whether a person *has ever been told they had a "
        "heart attack*. It does not predict validated future heart attack risk. "
        "If you have chest pain or other symptoms, seek medical help immediately."
    )

    # ---------- 9. Input section: the 19 features ----------
    st.header("📝 Patient Information")
    st.write("Fill in the details below. Move your mouse over the ? icon to see an explanation.")

    input_data = {}   # will hold one answer per feature
    for title, field_columns in input_groups:
        st.subheader(title)
        for column, fields in zip(st.columns(len(field_columns)), field_columns):
            with column:
                for field, label in fields:
                    if field == "BMI":
                        value = st.number_input(label, min_value=12.0, max_value=100.0,
                                                value=27.0, step=0.1, key="answer_" + field,
                                                help=help_text.get(field))
                    elif field == "SleepHours":
                        value = st.slider(label, min_value=1, max_value=24, value=7,
                                          key="answer_" + field, help=help_text.get(field))
                        value = float(value)   # float, same type as in training
                    else:
                        options = choices[field]
                        default = defaults.get(field, "No")
                        selected = options.index(default) if default in options else 0
                        value = st.selectbox(label, options, index=selected,
                                             key="answer_" + field, help=help_text.get(field))
                    input_data[field] = [value]   # one-item list, ready for a DataFrame

    # ---------- 10. Prediction ----------
    st.markdown("---")
    submitted = st.button("🔍 Predict")

    if submitted:
        st.session_state.pop("prediction_result", None)
        try:
            result, probability = predict_heart_attack(model, input_data)
            # Keep the result so it stays on screen when other widgets are used
            st.session_state["prediction_result"] = (result, probability, input_data)
        except (ValueError, KeyError):
            st.error("These answers could not be processed. Check that the saved model matches this app.")

    if "prediction_result" in st.session_state:
        result, probability, submitted_data = st.session_state["prediction_result"]
        st.header("📊 Prediction Result")
        st.caption("Based on your last submitted answers. Submit again after making changes.")
        col1, col2 = st.columns(2)
        with col1:
            if result == "Yes":
                st.warning("The model predicts a history of heart attack.")
            else:
                st.info("The model predicts no history of heart attack.")
        with col2:
            st.metric("Model-estimated probability of 'Yes'", f"{probability * 100:.1f}%")
            st.progress(probability)
        st.info(
            "How to read this: The model predicts Yes when the probability is "
            "above 50%. During training, 'Yes' cases were given 3 times more weight "
            "(class_weight = {'No': 1, 'Yes': 3}) to catch more cases. Because of this, "
            "the probability is a model score, not a real medical risk percentage. "
            "Please talk to a doctor about your actual heart health."
        )
        with st.expander("Review submitted answers"):
            st.dataframe(pd.DataFrame({
                "Question": [label for field, label in FIELDS],
                "Answer": [str(submitted_data[field][0]) for field, label in FIELDS]
            }), hide_index=True)

    # ---------- 11. Model performance ----------
    st.markdown("---")
    st.header("📈 Model Performance")
    st.write("Results of the tuned Logistic Regression model on the test set "
             "(20% of the data the model never saw during training). "
             "Precision, recall and F1-score are for the 'Yes' (heart attack) class.")
    try:
        metrics = joblib.load(MODEL_PATH.with_name("model_metrics.pkl"))
        for column, name in zip(st.columns(4), ["Accuracy", "Precision", "Recall", "F1-score"]):
            column.metric(name, f"{metrics[name] * 100:.1f}%")
    except (OSError, ValueError, KeyError, ImportError):
        st.info("Model performance scores are unavailable. Save model_metrics.pkl from the evaluation cell.")
    with st.expander("ℹ️ What do these scores mean?"):
        st.markdown("""
    Accuracy: the percentage of all predictions that were correct.
    Only about 6% of people in the data reported a heart attack, so a model that always says "No"
    would still get about 94% accuracy. That is why accuracy alone is misleading here.
    Precision: of all people the model predicted as "Yes", how many really reported a heart attack.
    Recall (Sensitivity): of all people who really reported a heart attack, how many the model found.
    In medicine this is very important, because a missed case (a false negative) is the most serious error.
    F1-score: a balance between precision and recall. It was used to choose the best model
    during tuning (GridSearchCV).
    """)
    # ---------- 13. Footer ----------
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #5D6D7E;'>Heart Attack Prediction App • "
    "Educational Machine Learning Project • Built with Streamlit and scikit-learn</p>",
    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
