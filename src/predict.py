import joblib
import pandas as pd

MODELS_DIR = "models"


# Load artifacts
model = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")


def prepare_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Convert categorical variables
    df = pd.get_dummies(df)

    # Align with training features
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict(input_dict):
    df = prepare_input(input_dict)

    # Scale input
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return prediction, probability