"""
UNIFIED PITCH DASHBOARD - Prakasam Police AI Hackathon 2026
Track: Cybercrime Detection & Digital Fraud

Run with:  streamlit run dashboard.py

Brings all 5 focus-area modules into one interactive demo for your live
pitch at Command HQ, instead of running 5 separate terminal scripts.
"""

import streamlit as st

import fraud_detection as fraud
import phishing_detection as phishing
import complaint_classification as complaints
import fake_identity_detection as identity
import threat_monitoring as threat

st.set_page_config(page_title="Digital Policing AI Suite", layout="wide")

st.title("🛡️ Digital Policing AI Suite")
st.caption("Prakasam Police AI Hackathon 2026 — Cybercrime Detection & Digital Fraud")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Financial Fraud",
    "🎣 Phishing & Scam",
    "📋 Complaint Routing",
    "🕵️ Fake Identity",
    "📡 Threat Monitoring",
])

# ---------- TAB 1: Financial Fraud ----------
with tab1:
    st.subheader("Financial Fraud Detection")
    st.write("Random Forest trained on transaction behaviour (amount, time, location jump, device change).")
    if st.button("Train model", key="fraud_train"):
        with st.spinner("Training..."):
            model, features = fraud.train_fraud_model()
        st.session_state["fraud_model"] = (model, features)
        st.success("Model trained. Score a transaction below.")

    if "fraud_model" in st.session_state:
        model, features = st.session_state["fraud_model"]
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount (₹)", value=15000)
        hour = c2.slider("Hour of day", 0, 23, 2)
        txn_count = c3.number_input("Txns in last 1hr", value=5)
        c4, c5, c6 = st.columns(3)
        distance = c4.number_input("Distance from last txn (km)", value=250)
        new_device = c5.checkbox("New device?", value=True)
        new_benef = c6.checkbox("New beneficiary?", value=True)

        if st.button("Score transaction"):
            txn = {
                "amount": amount, "hour_of_day": hour, "txn_count_last_1hr": txn_count,
                "distance_from_last_txn_km": distance,
                "is_new_device": int(new_device), "is_new_beneficiary": int(new_benef),
            }
            label, prob = fraud.score_transaction(model, features, txn)
            st.metric("Verdict", label, delta=f"{prob*100:.1f}% fraud probability")

# ---------- TAB 2: Phishing ----------
with tab2:
    st.subheader("Phishing & Scam Message Detection")
    st.write("TF-IDF + Linear SVM classifying SMS / email / WhatsApp text.")
    if st.button("Train model", key="phish_train"):
        model, vectorizer = phishing.train_phishing_model()
        st.session_state["phish_model"] = (model, vectorizer)
        st.success("Model trained.")

    if "phish_model" in st.session_state:
        model, vectorizer = st.session_state["phish_model"]
        msg = st.text_area("Paste a message to check", "Your bank account is suspended. Verify now at http://fake-bank-link.com")
        if st.button("Classify message"):
            label, score = phishing.classify_message(model, vectorizer, msg)
            st.metric("Verdict", label, delta=f"SVM score: {score}")

# ---------- TAB 3: Complaint Routing ----------
with tab3:
    st.subheader("Cyber Complaint Auto-Classification")
    st.write("Naive Bayes routes free-text complaints to the correct investigation desk.")
    if st.button("Train model", key="complaint_train"):
        model, vectorizer = complaints.train_complaint_classifier()
        st.session_state["complaint_model"] = (model, vectorizer)
        st.success("Model trained.")

    if "complaint_model" in st.session_state:
        model, vectorizer = st.session_state["complaint_model"]
        text = st.text_area("Paste complaint text", "Someone used my photo to make a fake account and is scamming my relatives")
        if st.button("Classify complaint"):
            category, confidence = complaints.classify_complaint(model, vectorizer, text)
            st.metric("Routed to desk", category, delta=f"confidence: {confidence}")

# ---------- TAB 4: Fake Identity ----------
with tab4:
    st.subheader("Fake Identity / Profile Detection")
    st.write("Decision Tree over account metadata — interpretable rules, great for explaining to officers.")
    if st.button("Train model", key="identity_train"):
        model, features = identity.train_fake_identity_model()
        st.session_state["identity_model"] = (model, features)
        st.success("Model trained.")

    if "identity_model" in st.session_state:
        model, features = st.session_state["identity_model"]
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Account age (days)", value=4)
        followers = c2.number_input("Followers", value=850)
        following = c3.number_input("Following", value=12)
        c4, c5, c6 = st.columns(3)
        has_photo = c4.checkbox("Has profile photo?", value=False)
        posts = c5.number_input("Post count", value=1)
        bio_len = c6.number_input("Bio length (chars)", value=0)

        if st.button("Check profile"):
            profile = {
                "account_age_days": age, "followers": followers, "following": following,
                "follow_ratio": followers / (following + 1),
                "has_profile_photo": int(has_photo), "posts_count": posts, "bio_length": bio_len,
            }
            verdict, prob = identity.check_profile(model, features, profile)
            st.metric("Verdict", verdict, delta=f"{prob*100:.1f}% fake probability")

# ---------- TAB 5: Threat Monitoring ----------
with tab5:
    st.subheader("Digital Threat Monitoring")
    st.write("Simulated live feed with rolling anomaly detection (z-score) to catch threat spikes early.")
    if st.button("Run 24-hour simulation"):
        feed = threat.simulate_feed()
        hourly, alerts = threat.detect_hourly_anomalies(feed)
        st.line_chart(hourly)
        if len(alerts) == 0:
            st.info("No anomalies detected in this simulation window.")
        else:
            for hour, score in alerts.items():
                st.error(f"⚠️ Hour {hour}: threat_score={score} — possible coordinated activity / emerging threat")

st.divider()
st.caption("Prototype built for Prakasam Police AI Hackathon 2026 · Models trained on synthetic data — swap in real (anonymised) datasets before production deployment.")