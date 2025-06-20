# ==============================================================================
# fraud_model_trainer.py
#
# This script performs the end-to-end process of training the fraud detection model.
# It handles data loading, preprocessing, feature engineering, model training,
# evaluation, and saving the final model pipeline.
# ==============================================================================

# 1. Imports and Setup
# ==============================================================================
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import kagglehub

print("--- Fraud Detection Model Training Script ---")

# 2. Data Loading
# ==============================================================================
print("\n[1/6] Downloading dataset from Kaggle Hub...")
try:
    path = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")
    dataset_path = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(dataset_path)
    print("Dataset downloaded and loaded successfully.")
except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    exit()

# 3. Data Cleaning and Feature Engineering
# ==============================================================================
print("\n[2/6] Performing data cleaning and feature engineering...")
df_cleaned = df.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest'])

# Feature Engineering
df_cleaned['hour_of_day'] = df_cleaned['step'] % 24
df_cleaned['balance_diff_orig'] = df_cleaned['oldbalanceOrg'] - df_cleaned['newbalanceOrig']
df_cleaned['balance_diff_dest'] = df_cleaned['newbalanceDest'] - df_cleaned['oldbalanceDest']
df_cleaned['amount_log'] = np.log1p(df_cleaned['amount'])

# Drop original columns
df_cleaned = df_cleaned.drop(columns=['step', 'amount'])
print("Feature engineering complete.")

# 4. Data Preprocessing and Splitting
# ==============================================================================
print("\n[3/6] Splitting data and defining preprocessing steps...")
X = df_cleaned.drop('isFraud', axis=1)
y = df_cleaned['isFraud']

# Identify categorical and numerical features for the preprocessor
categorical_features = ['type']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# 5. Model Training Pipeline
# ==============================================================================
print("\n[4/6] Building and training the model pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the model pipeline with SMOTE and XGBoost
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 6. Model Evaluation and Saving
# ==============================================================================
print("\n[5/6] Evaluating the trained model...")
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
print(f"--- Model Evaluation on Test Set ---")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
print("------------------------------------")

print("\n[6/6] Saving the final model pipeline...")
model_filename = 'fraud_detection_pipeline.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model_pipeline, file)

print(f"\nModel pipeline saved successfully to '{model_filename}'")
print("--- Script Finished ---")