import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --- MODEL LOADING ---
# NOTE: Ensure this Hugging Face repo contains the model trained on the Tourism data!
model_path = hf_hub_download(repo_id="shishirtiwari/tourism-project", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# --- STREAMLIT UI ---
st.title("🗺️ Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package (**ProdTaken**).
Please enter the customer and interaction details below to get a prediction.
""")

# --- USER INPUT SECTION ---

st.subheader("👤 Customer and Demographic Data")

# Demographic Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
designation = st.text_input("Designation (e.g., Manager, Senior Engineer)", "Executive")
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=30000.0, step=1000.0)

# Trip/Preference Inputs
city_tier = st.selectbox("City Tier", [1, 2, 3])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_trips = st.number_input("Number of Trips Annually", min_value=0, value=3)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, value=0)
num_adults = st.number_input("Number of Persons Visiting (Adults)", min_value=1, value=2) # Assuming this is 'NumberOfPersonVisiting' and represents adults

st.subheader("📞 Interaction Data")

contact_type = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_followups = st.number_input("Number of Follow-ups", min_value=0, value=2)
pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=1, value=15)


# --- ASSEMBLE INPUT DATAFRAME ---
# NOTE: The column names here MUST EXACTLY MATCH the names and order used 
# when training the model, including case and spelling.

input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': contact_type,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_adults,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': pitch_duration
}])


# --- PREDICTION LOGIC ---
if st.button("Predict Purchase Likelihood"):
    # Ensure the model outputs a probability if available
    try:
        # Predict probability of purchase (class 1)
        prediction_proba = model.predict_proba(input_data)[:, 1][0]
        prediction_class = model.predict(input_data)[0]
    except AttributeError:
        # Fallback if the model only has predict()
        prediction_class = model.predict(input_data)[0]
        prediction_proba = None # Cannot show probability

    st.subheader("Prediction Result:")
    
    if prediction_class == 1:
        st.success("The model predicts: **Package WILL BE Purchased!**")
    else:
        st.info("The model predicts: **Package will NOT be purchased.**")
        
    if prediction_proba is not None:
        st.write(f"Confidence Score (Probability of Purchase): **{prediction_proba:.2f}**")
