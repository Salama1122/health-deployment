from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "heart_attack_model.pkl"


@st.cache_resource
def load_model(path, modified_time):
    # The modification time refreshes the cache after the model is replaced.
    return joblib.load(path)


GROUPS = [
    ("01", "About you", "Start with your everyday profile.", [
        ("Sex", "Sex"), ("AgeCategory", "Age group"),
        ("BMI", "Body mass index (BMI)"), ("SleepHours", "Sleep per night (hours)"),
        ("GeneralHealth", "General health")]),
    ("02", "Everyday habits", "Choose the answers that best describe you.", [
        ("PhysicalActivities", "Physical activities"), ("SmokerStatus", "Smoking status"),
        ("ECigaretteUsage", "E-cigarette use"), ("AlcoholDrinkers", "Alcohol consumption")]),
    ("03", "Health history", "Select your reported history for each question.", [
        ("HadAngina", "History of angina"), ("HadAsthma", "History of asthma"),
        ("HadCOPD", "History of COPD"),
        ("HadDepressiveDisorder", "History of depressive disorder"),
        ("CovidPos", "Positive COVID-19 test")]),
    ("04", "Checkups & screening", "A few final details from the survey.", [
        ("LastCheckupTime", "Last routine checkup"), ("ChestScan", "Chest scan"),
        ("HIVTesting", "HIV testing"), ("TetanusLast10Tdap", "Tetanus shot in the past 10 years"),
        ("HighRiskLastYear", "High-risk activity in the past year (survey field)")]),
]


def reset_answers():
    for key in list(st.session_state):
        if key.startswith("answer_") or key == "prediction_result":
            del st.session_state[key]


def predict_heart_attack(model, input_data):
    frame = pd.DataFrame(input_data).reindex(columns=model.feature_names_in_)
    return str(model.predict(frame)[0])


def main():
    st.set_page_config(page_title="Heartwise | Heart history explorer", page_icon="?", layout="wide")
    st.markdown("""
    <style>
    .stApp {background: #f5f8fa; color: #183641;}
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {background: linear-gradient(120deg, #103c48, #126b70); border-radius: 24px;
           padding: 38px; color: white; margin: 12px 0 26px;}
    .hero h1 {color: white; font-size: clamp(2rem, 5vw, 3.3rem); letter-spacing: -1.5px;}
    .hero p {color: #d7efec; max-width: 640px; font-size: 1.05rem;}
    .eyebrow {text-transform: uppercase; letter-spacing: 2px; font-size: .75rem; font-weight: 700;}
    .badge {display: inline-block; background: #f9b9a5; color: #263c43; padding: 6px 14px;
            border-radius: 30px; font-size: .8rem; font-weight: 600; margin-top: 8px;}
    .step {font-size: .78rem; font-weight: 700; color: #087f80; letter-spacing: 1px;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
    [data-testid="stForm"] {background: white; border-radius: 20px;}
    div[data-testid="stFormSubmitButton"] button {background: #107a7b; color: white;
        border: none; border-radius: 12px; padding: .7rem 1.5rem; font-weight: 600;}
    div[data-testid="stFormSubmitButton"] button:hover {background: #095f60; color: white;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("**? HEARTWISE** ? A machine learning explorer")
    st.markdown("""
    <section class="hero">
      <div class="eyebrow">Your answers. A model's perspective.</div>
      <h1>A clearer view of<br>your heart history.</h1>
      <p>Explore how a trained model classifies reported heart attack history
      using lifestyle and health survey answers.</p>
      <span class="badge">19 questions &nbsp; ? &nbsp; 4 simple sections</span>
    </section>
    """, unsafe_allow_html=True)

    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, ImportError):
        st.error("The prediction model could not be loaded. Check the model file and installed dependencies.")
        st.stop()

    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_columns = next(columns for name, _, columns in preprocessor.transformers_ if name == "cat")
    choices = {column: list(values) for column, values in zip(cat_columns, encoder.categories_)}

    with st.expander("How to read your result"):
        st.write("The model returns Yes or No for a reported history of heart attack. "
                 "It can make mistakes. It does not predict a future event or diagnose a current one.")
        st.write("Review every answer before submitting. The starting values are examples, not a personal profile.")

    left, right = st.columns([3, 1])
    with left:
        st.subheader("Build your profile")
        st.caption("Review all four sections, then select Predict.")
    with right:
        st.button("Reset answers", on_click=reset_answers, use_container_width=True)

    input_data = {}
    with st.form("heart_profile"):
        for number, title, description, fields in GROUPS:
            st.markdown(f'<div class="step">SECTION {number}</div>', unsafe_allow_html=True)
            st.subheader(title)
            st.caption(description)
            columns = st.columns(2)
            for index, (field, label) in enumerate(fields):
                with columns[index % 2]:
                    if field == "BMI":
                        value = st.number_input(label, min_value=1.0, max_value=150.0,
                                                value=25.0, step=0.1, key="answer_" + field)
                    elif field == "SleepHours":
                        value = st.number_input(label, min_value=0, max_value=24,
                                                value=7, key="answer_" + field)
                    else:
                        options = choices[field]
                        defaults = {"Sex": "Female", "AgeCategory": "Age 55 to 59",
                                    "GeneralHealth": "Good", "SmokerStatus": "Never smoked",
                                    "ECigaretteUsage": "Never used e-cigarettes in my entire life",
                                    "LastCheckupTime": "Within past year (anytime less than 12 months ago)"}
                        default = defaults.get(field, "No")
                        selected = options.index(default) if default in options else 0
                        value = st.selectbox(label, options, index=selected, key="answer_" + field)
                    input_data[field] = [value]
            st.divider()
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        st.session_state.pop("prediction_result", None)
        try:
            result = predict_heart_attack(model, input_data)
            st.session_state["prediction_result"] = (result, input_data)
        except (ValueError, KeyError):
            st.error("These answers could not be processed. Check that the saved model matches this app.")

    if "prediction_result" in st.session_state:
        result, submitted_data = st.session_state["prediction_result"]
        with st.container(border=True):
            st.subheader("Your model result")
            st.caption("Based on your last submitted answers. Submit again after making changes.")
            if result == "Yes":
                st.warning("The model predicts a history of heart attack.")
            else:
                st.info("The model predicts no history of heart attack.")
            with st.expander("Review submitted answers"):
                st.dataframe(pd.DataFrame({"Question": [label for _, _, _, fields in GROUPS for _, label in fields],
                                          "Answer": [str(submitted_data[field][0]) for _, _, _, fields in GROUPS for field, _ in fields]}),
                             hide_index=True, use_container_width=True)
    else:
        st.info("Your result will appear here after you submit your profile.")
    st.caption("This prediction is not a medical diagnosis and does not determine whether you are currently having a heart attack.")
    st.markdown("---")
    st.caption("HEARTWISE ? Built for learning and exploration")


if __name__ == "__main__":
    main()
