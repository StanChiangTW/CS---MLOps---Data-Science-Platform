import os
import joblib
import pandas as pd
from preprocessing_a import preprocess_data  # Assuming preprocess_data is defined elsewhere
from model_training import train_svm_classifier, train_rfc_classifier, train_xgboost_classifier, train_lgbm_classifier  # Assuming these functions are defined elsewhere

# Define generalized file path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILE_PATH = os.path.join(BASE_DIR, "data", "BankChurners.csv")

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data(FILE_PATH)

# Save X_test and y_test separately
X_test.to_csv(os.path.join(BASE_DIR, "data", "X_test.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "data", "y_test.csv"), index=False)

# Define model IDs
svm_model_id = "svm_model"
rf_model_id = "rf_model"
xgb_model_id = "xgb_model"
lgbm_model_id = "lgbm_model"

# Train models
svm_model, svm_metadata = train_svm_classifier(X_train, y_train, svm_model_id)
rf_model, rf_metadata = train_rfc_classifier(X_train, y_train, rf_model_id)
xgb_model, xgb_metadata = train_xgboost_classifier(X_train, y_train, xgb_model_id)
lgbm_model, lgbm_metadata = train_lgbm_classifier(X_train, y_train, lgbm_model_id)

# Ensure models directory exists
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Save trained models
joblib.dump(svm_model, os.path.join(models_dir, "svm_model.pkl"))
joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
joblib.dump(lgbm_model, os.path.join(models_dir, "lgbm_model.pkl"))

print("All models trained and saved successfully!")
