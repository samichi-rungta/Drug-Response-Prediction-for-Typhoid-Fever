# Improved Typhoid Drug Response Prediction with Extended Evaluation Metrics

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)

# Step 1: Load Data
df = pd.read_csv('DRP-for-Typhoid-Fever.csv')  # Change this to your actual dataset path

# Step 2: Clean 'Treatment Duration'
df['Treatment Duration'] = df['Treatment Duration'].str.extract(r'(\d+)').astype(float)

# Step 3: Encode Categorical Variables
categorical_cols = ['Gender', 'Symptoms Severity', 'Blood Culture Bacteria', 
                    'Urine Culture Bacteria', 'Current Medication']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Feature Selection
X = df.drop(['Patient ID', 'Treatment Outcome'], axis=1)
y = df['Treatment Outcome'].map({'Unsuccessful': 0, 'Successful': 1})  # Map outcome to 0/1

# Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Model Building with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Step 8: Evaluation
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # For ROC AUC

print("\nBest Parameters Found:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Random chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 9: Save the Model
joblib.dump(best_rf, 'improved_typhoid_model.pkl')
joblib.dump(label_encoders, 'improved_label_encoders.pkl')
joblib.dump(scaler, 'improved_scaler.pkl')

print("\n✅ Model Saved Successfully!")
