"""
FOCUS AREA 1: Financial Fraud Detection
----------------------------------------
Flags suspicious bank/UPI transactions using a Random Forest classifier
trained on transaction behaviour features (amount, time, frequency, location jump).

In a real deployment you'd swap the synthetic data generator for real
(anonymised) transaction logs from banks / UPI providers.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)


def generate_transaction_data(n=5000):
    """Creates synthetic transactions. ~5% are fraudulent."""
    data = {
        "amount": np.random.exponential(scale=2000, size=n),
        "hour_of_day": np.random.randint(0, 24, n),
        "txn_count_last_1hr": np.random.poisson(1.5, n),
        "distance_from_last_txn_km": np.random.exponential(scale=15, size=n),
        "is_new_device": np.random.binomial(1, 0.08, n),
        "is_new_beneficiary": np.random.binomial(1, 0.15, n),
    }
    df = pd.DataFrame(data)

    # Rule-based ground truth generation (simulates real fraud patterns)
    fraud_score = (
        (df["amount"] > 8000).astype(int) * 2
        + (df["hour_of_day"].isin([0, 1, 2, 3, 4])).astype(int) * 2
        + (df["txn_count_last_1hr"] > 4).astype(int) * 2
        + (df["distance_from_last_txn_km"] > 100).astype(int) * 2
        + df["is_new_device"] * 2
        + df["is_new_beneficiary"] * 1
    )
    df["is_fraud"] = (fraud_score >= 6).astype(int)
    # add a little noise so it's not trivially separable
    flip_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df.loc[flip_idx, "is_fraud"] = 1 - df.loc[flip_idx, "is_fraud"]
    return df


def train_fraud_model():
    df = generate_transaction_data()
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("=== Financial Fraud Detection: Model Report ===")
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop signals driving fraud predictions:\n", importances)
    return model, X.columns.tolist()


def score_transaction(model, feature_names, txn: dict):
    """Score a single live transaction. Returns risk label + probability."""
    row = pd.DataFrame([txn])[feature_names]
    prob = model.predict_proba(row)[0][1]
    label = "HIGH RISK - FLAG FOR REVIEW" if prob > 0.5 else "LOW RISK"
    return label, round(float(prob), 3)


if __name__ == "__main__":
    model, features = train_fraud_model()

    # Example: a suspicious late-night transaction from a new device
    sample_txn = {
        "amount": 15000,
        "hour_of_day": 2,
        "txn_count_last_1hr": 5,
        "distance_from_last_txn_km": 250,
        "is_new_device": 1,
        "is_new_beneficiary": 1,
    }
    label, prob = score_transaction(model, features, sample_txn)
    print(f"\nSample live transaction -> {label} (fraud probability: {prob})")