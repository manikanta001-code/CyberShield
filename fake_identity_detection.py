"""
FOCUS AREA 4: Fake Identity Detection
-------------------------------------
Estimates how likely a social-media / registration profile is FAKE, using
account metadata (age, followers/following ratio, profile completeness,
posting behaviour) with a Decision Tree - deliberately chosen because it's
interpretable, and you can literally show the police the decision rules
during your pitch (e.g. "account age < 7 days AND no profile photo -> fake").
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(7)


def generate_profile_data(n=3000):
    account_age_days = np.random.exponential(scale=300, size=n).astype(int)
    followers = np.random.exponential(scale=200, size=n).astype(int)
    following = np.random.exponential(scale=150, size=n).astype(int)
    has_profile_photo = np.random.binomial(1, 0.75, n)
    posts_count = np.random.exponential(scale=40, size=n).astype(int)
    bio_length = np.random.exponential(scale=30, size=n).astype(int)

    df = pd.DataFrame({
        "account_age_days": account_age_days,
        "followers": followers,
        "following": following,
        "follow_ratio": followers / (following + 1),
        "has_profile_photo": has_profile_photo,
        "posts_count": posts_count,
        "bio_length": bio_length,
    })

    fake_score = (
        (df["account_age_days"] < 15).astype(int) * 3
        + (df["follow_ratio"] < 0.05).astype(int) * 2
        + (df["has_profile_photo"] == 0).astype(int) * 3
        + (df["posts_count"] < 3).astype(int) * 2
        + (df["bio_length"] < 5).astype(int) * 1
    )
    df["is_fake"] = (fake_score >= 6).astype(int)
    return df


def train_fake_identity_model():
    df = generate_profile_data()
    X = df.drop(columns=["is_fake"])
    y = df["is_fake"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("=== Fake Identity Detection: Model Report ===")
    print(classification_report(y_test, preds, target_names=["Genuine", "Fake"]))

    print("\nHuman-readable decision rules (great for the pitch demo):")
    print(export_text(model, feature_names=list(X.columns)))

    return model, X.columns.tolist()


def check_profile(model, feature_names, profile: dict):
    row = pd.DataFrame([profile])[feature_names]
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    verdict = "LIKELY FAKE - FLAG FOR VERIFICATION" if pred == 1 else "LIKELY GENUINE"
    return verdict, round(float(prob), 3)


if __name__ == "__main__":
    model, features = train_fake_identity_model()

    suspicious_profile = {
        "account_age_days": 4,
        "followers": 850,
        "following": 12,
        "follow_ratio": 850 / 13,
        "has_profile_photo": 0,
        "posts_count": 1,
        "bio_length": 0,
    }
    verdict, prob = check_profile(model, features, suspicious_profile)
    print(f"\nSample profile check -> {verdict} (fake probability: {prob})")