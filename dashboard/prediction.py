import pandas as pd
import joblib
import os

def load_prediction_artifacts():
    """Locates the models folder by tracking the project directory structure explicitly."""
    # Find the folder we are running in
    current_path = os.path.abspath(__file__)
    
    # Climb up until we hit our root project directory
    while os.path.basename(current_path) != "Customer-Churn-Dashboard":
        parent = os.path.dirname(current_path)
        if parent == current_path: # Breakout fallback if not found
            break
        current_path = parent
        
    # Construct the definitive absolute path directly to your models folder
    model_path = os.path.join(current_path, 'models', 'churn_model.pkl')
    preprocessor_path = os.path.join(current_path, 'models', 'preprocessor.pkl')
    
    # Let's print out exactly where it's looking to the console for easy debugging
    print(f"[DEBUG] Attempting to load model from: {model_path}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict_single_customer(input_data):
    model, preprocessor = load_prediction_artifacts()
    input_df = pd.DataFrame([input_data])
    processed_input = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_input)[0][1]
    
    if prob >= 0.70:
        risk_tier = "High Risk"
    elif prob >= 0.40:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"
        
    return risk_tier, prob