"""
Streamlit front-end for the Wellness Tourism Package predictor.
Loads the trained model from the Hugging Face model hub and serves predictions.
"""

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME = os.getenv("HF_USERNAME", "prudvikrishna")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"
MODEL_FILE = "best_tourism_model.joblib"


@st.cache_resource
def load_model():
    """Download and cache the model from the HF model hub."""
    path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILE,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),  # optional for public repos
    )
    return joblib.load(path)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Purchase Predictor")
st.write(
    "Predict whether a customer is likely to purchase the **Wellness Tourism "
    "Package** before the sales team contacts them."
)

model = load_model()

# --------------------------------------------------------------------------- #
# Collect inputs                                                               #
# --------------------------------------------------------------------------- #
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=0.0, value=15.0)
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number Of Persons Visiting", min_value=1, value=3)
    num_followups = st.number_input("Number Of Followups", min_value=0.0, value=3.0)
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number Of Trips (per year)", min_value=0.0, value=3.0)
    passport = st.selectbox("Has Passport", [0, 1])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns Car", [0, 1])
    num_children_visiting = st.number_input("Number Of Children Visiting", min_value=0.0, value=1.0)
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0)

# --------------------------------------------------------------------------- #
# Build a single-row dataframe in the SAME schema used at training time        #
# --------------------------------------------------------------------------- #
input_df = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    if pred == 1:
        st.success(f"✅ Likely to PURCHASE the package (probability = {proba:.2%})")
    else:
        st.info(f"❌ Unlikely to purchase the package (probability = {proba:.2%})")
    st.caption("Threshold = 0.50")
