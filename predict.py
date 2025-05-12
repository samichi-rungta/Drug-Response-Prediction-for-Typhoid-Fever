# Predict Treatment Outcome for New Patient Data

import pandas as pd
import numpy as np
import joblib

# Step 1: Load Saved Model, Encoders, and Scaler
model = joblib.load('improved_typhoid_model.pkl')
label_encoders = joblib.load('improved_label_encoders.pkl')
scaler = joblib.load('improved_scaler.pkl')

# Step 2: Input New Patient Information
# Example new patient data (you can replace these values)
new_patient = {
    'Age': 30,
    'Gender': 'Male',
    'Symptoms Severity': 'High',
    'Hemoglobin (g/dL)': 12.5,
    'Platelet Count': 250000,
    'Blood Culture Bacteria': 'Salmonella typhi',
    'Urine Culture Bacteria': 'None',
    'Calcium (mg/dL)': 9.2,
    'Potassium (mmol/L)': 4.5,
    'Current Medication': 'Azithromycin',
    'Treatment Duration': 14  # Already numeric
}

# Step 3: Convert to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Step 4: Encode Categorical Columns
for col in ['Gender', 'Symptoms Severity', 'Blood Culture Bacteria', 'Urine Culture Bacteria', 'Current Medication']:
    if col in new_patient_df.columns:
        le = label_encoders[col]
        # Handle unseen categories safely
        if new_patient_df[col].values[0] in le.classes_:
            new_patient_df[col] = le.transform(new_patient_df[col])
        else:
            # If unseen, add to classes_ temporarily
            le.classes_ = np.append(le.classes_, new_patient_df[col].values[0])
            new_patient_df[col] = le.transform(new_patient_df[col])

# Step 5: Scale Features
new_patient_scaled = scaler.transform(new_patient_df)

# Step 6: Predict
prediction = model.predict(new_patient_scaled)[0]
prediction_proba = model.predict_proba(new_patient_scaled)[0][1]

# Step 7: Output
outcome = 'Successful' if prediction == 1 else 'Unsuccessful'
print(f"\n🩺 Predicted Treatment Outcome: {outcome}")
print(f"📈 Success Probability: {prediction_proba:.2%}")
