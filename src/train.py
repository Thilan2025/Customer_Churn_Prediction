import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocess import preprocess_data


DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = "models"


def evaluate_model(name, model, x_test, y_test, y_pred, y_prob=None):
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        results["ROC-AUC"] = roc_auc_score(y_test, y_prob)

    return results


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = preprocess_data(DATA_PATH)

    if "Churn_Yes" in df.columns:
        target_col = "Churn_Yes"
    elif "Churn" in df.columns:
        target_col = "Churn"
    else:
        raise ValueError("Target column not found. Expected 'Churn_Yes' or 'Churn'.")

    x = df.drop(columns=[target_col])
    y = df[target_col]

    # Save feature columns for later use in the app
    joblib.dump(list(x.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    results = []

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(x_train_scaled, y_train)
    y_pred_log = log_model.predict(x_test_scaled)
    y_prob_log = log_model.predict_proba(x_test_scaled)[:, 1]

    results.append(
        evaluate_model(
            "Logistic Regression",
            log_model,
            x_test_scaled,
            y_test,
            y_pred_log,
            y_prob_log,
        )
    )
    joblib.dump(log_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(x_train, y_train)
    y_pred_dt = dt_model.predict(x_test)
    y_prob_dt = dt_model.predict_proba(x_test)[:, 1]

    results.append(
        evaluate_model(
            "Decision Tree",
            dt_model,
            x_test,
            y_test,
            y_pred_dt,
            y_prob_dt,
        )
    )
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "decision_tree.pkl"))

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)
    y_prob_rf = rf_model.predict_proba(x_test)[:, 1]

    results.append(
        evaluate_model(
            "Random Forest",
            rf_model,
            x_test,
            y_test,
            y_pred_rf,
            y_prob_rf,
        )
    )
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1-score", ascending=False)

    print("\nModel Comparison Results:\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest model based on F1-score: {best_model_name}")


if __name__ == "__main__":
    main()