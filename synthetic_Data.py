import pandas as pd
import numpy as np
from scipy.stats import norm
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define number of new synthetic samples
num_samples = 1000  # Adjust this number as needed

# Load existing dataset
file_path = "DRP-for-Typhoid-Fever.csv"  # Replace with your actual file path
existing_data = pd.read_csv(file_path)

# Ensure existing data has required columns
required_columns = [
    "Patient ID", "Age", "Gender", "Symptoms Severity", 
    "Hemoglobin (g/dL)", "Platelet Count", "Blood Culture Bacteria", 
    "Urine Culture Bacteria", "Calcium (mg/dL)", "Potassium (mmol/L)", 
    "Current Medication", "Treatment Duration", "Treatment Outcome"
]

# If dataset is empty or missing columns, initialize from scratch
if set(required_columns).issubset(existing_data.columns):
    last_patient_id = existing_data["Patient ID"].max() if not existing_data.empty else 1000
else:
    last_patient_id = 1000
    existing_data = pd.DataFrame(columns=required_columns)

# Generate synthetic Patient ID (continuing from last ID)
patient_ids = list(range(last_patient_id + 1, last_patient_id + num_samples + 1))

# Generate synthetic demographic details
ages = np.random.randint(5, 75, size=num_samples)  # Patients aged between 5-75 years
genders = [random.choice(["Male", "Female"]) for _ in range(num_samples)]
symptoms_severity = [random.choice(["Low", "Moderate", "High", "Severe"]) for _ in range(num_samples)]  # Categorical

# Generate synthetic numerical values
hemoglobin = np.round(norm.rvs(loc=13.5, scale=1.5, size=num_samples), 2)  # Normal range: 12-16 g/dL
platelet_count = np.round(norm.rvs(loc=250000, scale=50000, size=num_samples), 0)  # 150,000 - 450,000 /µL
calcium = np.round(norm.rvs(loc=9.5, scale=0.5, size=num_samples), 2)  # 8.5-10.5 mg/dL
potassium = np.round(norm.rvs(loc=4.0, scale=0.4, size=num_samples), 2)  # 3.5-5.1 mmol/L

# Generate categorical values
blood_culture_bacteria = [random.choice(["Salmonella Typhi", "Salmonella Paratyphi A", "Salmonella Paratyphi B", "Escherichia coli", "No Growth"]) for _ in range(num_samples)]
urine_culture_bacteria = [random.choice(["E. coli", "Klebsiella pneumoniae", "Proteus", "No Growth"]) for _ in range(num_samples)]
current_medication = [random.choice(["Ceftriaxone", "Azithromycin", "Ciprofloxacin", "Amoxicillin", "Meropenem"]) for _ in range(num_samples)]
treatment_duration = [f"{random.randint(5, 15)} days" for _ in range(num_samples)]  # Keeping "days" in text

# Treatment outcome logic
treatment_outcome = [
    "Successful" if (med in ["Ceftriaxone", "Azithromycin"] and bac not in ["No Growth"]) else "Unsuccessful"
    for med, bac in zip(current_medication, blood_culture_bacteria)
]

# Create DataFrame for new data
new_data = pd.DataFrame({
    "Patient ID": patient_ids,
    "Age": ages,
    "Gender": genders,  
    "Symptoms Severity": symptoms_severity,
    "Hemoglobin (g/dL)": hemoglobin,
    "Platelet Count": platelet_count,
    "Blood Culture Bacteria": blood_culture_bacteria,  
    "Urine Culture Bacteria": urine_culture_bacteria,  
    "Calcium (mg/dL)": calcium,
    "Potassium (mmol/L)": potassium,
    "Current Medication": current_medication,  
    "Treatment Duration": treatment_duration,
    "Treatment Outcome": treatment_outcome  
})

# Append new data to existing dataset
updated_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save updated dataset
updated_data.to_csv(file_path, index=False)
print(f"Added {num_samples} new synthetic records to {file_path}.")
