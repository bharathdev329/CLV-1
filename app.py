import pickle
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

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

# Route for the homepage
@app.route('/')
def home():
    return render_template('homepage.html')

# Route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        total_claim_amount = float(request.form['Total_Claim_Amount'])
        income = float(request.form['Income'])
        monthly_premium_auto = float(request.form['Monthly_Premium_Auto'])
        months_since_last_claim = float(request.form['Months_Since_Last_Claim'])
        months_since_policy_inception = float(request.form['Months_Since_Policy_Inception'])
        number_of_policies = int(request.form['Number_of_Policies'])
        state = request.form['State']
        policy_type = request.form['PolicyType']
        vehicle_class = request.form['VehicleClass']

        # Reshape the state, policy_type, and vehicle_class input to 2D array (required by TargetEncoder)
        state_encoded = state_encoder.transform(np.array([[state]]))[0]
        policy_type_encoded = policy_type_encoder.transform(np.array([[policy_type]]))[0]
        vehicle_class_encoded = vehicle_class_encoder.transform(np.array([[vehicle_class]]))[0]

        # Prepare features for scaling (only the features that MinMaxScaler was trained on)
        features_to_scale = [
            total_claim_amount, 
            income, 
            monthly_premium_auto, 
            months_since_last_claim, 
            months_since_policy_inception
        ]

        # Scale only the relevant features (scaler expects a 2D array)
        scaled_features = scaler.transform([features_to_scale])[0]

        # Create the final model input array
        model_input = np.array([
            scaled_features[0],  # Scaled Total_Claim_Amount
            scaled_features[1],  # Scaled Income
            scaled_features[2],  # Scaled Monthly_Premium_Auto
            scaled_features[3],  # Scaled Months_Since_Last_Claim
            scaled_features[4],  # Scaled Months_Since_Policy_Inception
            number_of_policies,      # Number of Policies (non-scaled)
            *state_encoded,           # Encoded State (non-scaled)
            *policy_type_encoded,     # Encoded PolicyType (non-scaled)
            *vehicle_class_encoded    # Encoded VehicleClass (non-scaled)
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(model_input)[0]

        # Return the result with prediction on the same page
        return render_template('prediction.html', prediction_text=f'Predicted CLV: ${prediction:.2f}')
    
    return render_template('prediction.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the developer page
@app.route('/developer')
def developer():
    return render_template('developer.html')

if __name__ == '__main__':
    app.run(debug=True)
