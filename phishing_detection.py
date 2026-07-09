"""
FOCUS AREA 2: Phishing & Scam Detection
-----------------------------------------
Classifies SMS / email / WhatsApp text messages as PHISHING or SAFE using
TF-IDF text features + a Linear SVM (the same SVM concepts you've been
studying - hyperplane separating two classes, just applied to text vectors).

Swap `SAMPLE_MESSAGES` for a real labelled dataset (e.g. Kaggle's
"SMS Spam Collection" or a scraped cybercrime-complaint corpus) for production use.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SAMPLE_MESSAGES = [
    ("Your KYC will expire today. Update immediately at http://bit.ly/kyc-verify or account blocked", 1),
    ("Congratulations! You won Rs 50,00,000 in lucky draw. Claim now: http://claim-prize.xyz", 1),
    ("Dear customer your ATM card is blocked. Share OTP to reactivate urgently", 1),
    ("URGENT: your electricity will be disconnected tonight. Pay now http://pay-bill-now.top", 1),
    ("Your parcel is on hold due to unpaid customs fee. Pay Rs 50 here http://track-parcel.fake", 1),
    ("Bank alert: unusual login detected. Verify identity here http://secure-bank-verify.co", 1),
    ("You have been selected for a work from home job paying 5000/day. Register: http://job-scam.link", 1),
    ("Hi, are we still meeting for lunch tomorrow at 1pm?", 0),
    ("Your Amazon order #45231 has been shipped and will arrive Friday", 0),
    ("Reminder: Team meeting rescheduled to 3 PM today", 0),
    ("Mom, I'll be home late tonight, don't wait for dinner", 0),
    ("Your OTP for login is 482913. Do not share this with anyone.", 0),
    ("Electricity bill of Rs 1240 is due on 15th. Pay via official app.", 0),
    ("Your college fee receipt has been generated. Download from student portal.", 0),
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)  # normalize links -> strong phishing signal
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def train_phishing_model():
    df = pd.DataFrame(SAMPLE_MESSAGES, columns=["text", "label"])
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC(C=1.0, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("=== Phishing & Scam Detection: Model Report ===")
    print(classification_report(y_test, preds, target_names=["Safe", "Phishing"], zero_division=0))

    return model, vectorizer


def classify_message(model, vectorizer, message: str):
    clean = clean_text(message)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    score = model.decision_function(vec)[0]  # distance from SVM hyperplane
    label = "PHISHING / SCAM" if pred == 1 else "SAFE"
    return label, round(float(score), 3)


if __name__ == "__main__":
    model, vectorizer = train_phishing_model()

    test_msgs = [
        "Your bank account is suspended. Verify now at http://fake-bank-link.com",
        "Hey, don't forget to bring the notes for tomorrow's class",
    ]
    for msg in test_msgs:
        label, score = classify_message(model, vectorizer, msg)
        print(f"\nMessage: {msg}\n-> {label} (SVM confidence score: {score})")