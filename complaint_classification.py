"""
FOCUS AREA 3: Cyber Complaint Classification
------------------------------------------------
Automatically routes an incoming free-text cybercrime complaint (like the ones
filed on cybercrime.gov.in) into the correct category, so it reaches the
right investigation desk faster instead of sitting in a manual triage queue.

Uses Multinomial Naive Bayes (fast, works well on short text, easy to explain
to non-technical police officers during the pitch).
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SAMPLE_COMPLAINTS = [
    ("Someone hacked my Instagram and is asking my friends for money", "Social Media / Identity Misuse"),
    ("My Facebook account was cloned and fake account is messaging my contacts", "Social Media / Identity Misuse"),
    ("Unknown person created a fake profile using my photos", "Social Media / Identity Misuse"),
    ("I transferred money to a fake investment app and now it's not responding", "Financial Fraud"),
    ("Rs 20000 was debited from my account without my knowledge via UPI", "Financial Fraud"),
    ("I was asked to pay advance fee for a loan that never came", "Financial Fraud"),
    ("I received a call saying my SIM will be blocked, then money was deducted", "Financial Fraud"),
    ("I got a message with a link saying I won a lottery and lost money after clicking", "Phishing / Scam Link"),
    ("Received fake courier delivery link asking for customs payment", "Phishing / Scam Link"),
    ("Someone is blackmailing me with my private photos", "Cyber Blackmail / Harassment"),
    ("I am getting continuous abusive messages from an unknown number", "Cyber Blackmail / Harassment"),
    ("My child is being bullied and threatened in an online game chat", "Cyber Blackmail / Harassment"),
    ("My laptop got locked with a message demanding bitcoin payment", "Ransomware / Malware"),
    ("A suspicious file I downloaded encrypted all my files", "Ransomware / Malware"),
    ("My computer is behaving abnormally after opening an email attachment", "Ransomware / Malware"),
]


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def train_complaint_classifier():
    df = pd.DataFrame(SAMPLE_COMPLAINTS, columns=["text", "category"])
    df["clean"] = df["text"].apply(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["category"], test_size=0.3, random_state=42
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("=== Cyber Complaint Classification: Model Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    return model, vectorizer


def classify_complaint(model, vectorizer, complaint_text: str):
    clean_txt = clean(complaint_text)
    vec = vectorizer.transform([clean_txt])
    category = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return category, round(float(proba), 3)


if __name__ == "__main__":
    model, vectorizer = train_complaint_classifier()

    new_complaint = "Someone used my Aadhaar photo to make a fake WhatsApp DP and is scamming my relatives"
    category, confidence = classify_complaint(model, vectorizer, new_complaint)
    print(f"\nNew complaint: {new_complaint}")
    print(f"-> Routed to desk: {category} (confidence: {confidence})")