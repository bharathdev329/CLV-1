import pickle
import numpy as np
import streamlit as st

# Load the ensemble model
with open('ensemble_model5.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoders for State, PolicyType, and VehicleClass individually
with open('Target_Encoder_state1.pkl', 'rb') as state_encoder_file:
    state_encoder = pickle.load(state_encoder_file)

with open('Target_Encoder_Policy_type1.pkl', 'rb') as policy_type_encoder_file:
    policy_type_encoder = pickle.load(policy_type_encoder_file)

with open('Target_Encoder_Vehicle Class1.pkl', 'rb') as vehicle_class_encoder_file:
    vehicle_class_encoder = pickle.load(vehicle_class_encoder_file)

# Load the MinMaxScaler
with open('minmax_scale2.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app interface
st.title("Customer Lifetime Value")

# Dropdown options for State, Policy Type, and Vehicle Class
state_options = ['Arizona', 'Nevada', 'California', 'Washington', 'Oregon']
policy_type_options = ['Personal Auto', 'Corporate Auto', 'Special Auto']
vehicle_class_options = ['Four-Door Car', 'Two-Door Car', 'SUV', 'Luxury SUV', 'Sports Car', 'Luxury Car']

# User inputs
total_claim_amount = st.text_input('Total Claim Amount', value='0')
income = st.text_input('Income', value='0')
monthly_premium_auto = st.text_input('Monthly Premium Auto', value='0')
months_since_last_claim = st.text_input('Months Since Last Claim', value='0')
months_since_policy_inception = st.text_input('Months Since Policy Inception', value='0')
number_of_policies = st.number_input('Number of Policies', min_value=1, value=1)

# Dropdown inputs
state = st.selectbox('State', state_options)
policy_type = st.selectbox('Policy Type', policy_type_options)
vehicle_class = st.selectbox('Vehicle Class', vehicle_class_options)

# Predict button
if st.button('Predict'):
    # Convert input to float or int
    total_claim_amount = float(total_claim_amount)
    income = float(income)
    monthly_premium_auto = float(monthly_premium_auto)
    months_since_last_claim = float(months_since_last_claim)
    months_since_policy_inception = float(months_since_policy_inception)

    # Encode categorical features
    state_encoded = state_encoder.transform(np.array([[state]]))[0]
    policy_type_encoded = policy_type_encoder.transform(np.array([[policy_type]]))[0]
    vehicle_class_encoded = vehicle_class_encoder.transform(np.array([[vehicle_class]]))[0]

    # Prepare features for scaling
    features_to_scale = [
        total_claim_amount,
        income,
        monthly_premium_auto,
        months_since_last_claim,
        months_since_policy_inception
    ]

    # Scale numeric features
    scaled_features = scaler.transform([features_to_scale])[0]

    # Combine scaled features with encoded categorical features
    model_input = np.array([
        scaled_features[0],  # Scaled Total_Claim_Amount
        scaled_features[1],  # Scaled Income
        scaled_features[2],  # Scaled Monthly_Premium_Auto
        scaled_features[3],  # Scaled Months_Since_Last_Claim
        scaled_features[4],  # Scaled Months_Since_Policy_Inception
        number_of_policies,  # Number of Policies
        *state_encoded,      # Encoded State
        *policy_type_encoded,  # Encoded Policy Type
        *vehicle_class_encoded  # Encoded Vehicle Class
    ]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(model_input)[0]

    # Display the result
    st.write(f'Predicted CLV: ${prediction:.2f}')