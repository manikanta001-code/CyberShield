# Digital Policing AI Suite
### Prakasam Police AI Hackathon 2026 — Track: Cybercrime Detection & Digital Fraud

A prototype covering all 5 required focus areas, each as a standalone
trainable module plus one combined Streamlit dashboard for the live pitch.

## Files

| File | Focus Area | Model Used |
|---|---|---|
| `fraud_detection.py` | Financial Fraud Detection | Random Forest |
| `phishing_detection.py` | Phishing & Scam Detection | TF-IDF + Linear SVM |
| `complaint_classification.py` | Cyber Complaint Classification | Naive Bayes (text) |
| `fake_identity_detection.py` | Fake Identity Detection | Decision Tree |
| `threat_monitoring.py` | Digital Threat Monitoring | Keyword scoring + z-score anomaly detection |
| `dashboard.py` | Combined UI for all 5 | Streamlit |

## Setup

```bash
pip install scikit-learn pandas numpy streamlit --break-system-packages
```

## Run individually (terminal, shows model reports + metrics)

```bash
python fraud_detection.py
python phishing_detection.py
python complaint_classification.py
python fake_identity_detection.py
python threat_monitoring.py
```

## Run the live pitch dashboard

```bash
streamlit run dashboard.py
```

This opens a browser UI with tabs for all 5 tools — train each model live
and score real inputs in front of the judges.

## ⚠️ Before you submit — read this

This is a **prototype skeleton**, not a finished submission. All the training
data right now is **synthetic** (generated with numpy/random rules) so the
code runs end-to-end and you have something to demo immediately. To make
this a genuinely strong hackathon entry, you should:

1. **Swap in real datasets** — e.g.:
   - Fraud: Kaggle "Credit Card Fraud Detection" dataset
   - Phishing: Kaggle "SMS Spam Collection" or "Phishing Website Dataset"
   - Complaints: scrape/anonymize sample complaint categories from public
     cybercrime awareness resources (never use real victim data without permission)
2. **Tune the models** — try `GridSearchCV` (you've already learned this)
   on the Random Forest / SVM to improve real-world accuracy.
3. **Add a proper explanation slide** in your pitch deck: problem →
   approach → model choice → why it's deployable → impact for police.
4. **Check the exact submission format** required by the organizers
   (GitHub link, PPT, live demo) — this wasn't fully visible on the site
   pages you shared, so confirm with Command HQ / the registration portal
   directly before July 10.

## Why these model choices

Each module intentionally uses a model you're already studying, so you can
explain the *how* confidently if judges ask:
- **Random Forest** (fraud) — handles mixed numeric signals, robust to noise
- **SVM** (phishing) — finds the best separating hyperplane between phishing
  vs. safe text after TF-IDF vectorization
- **Naive Bayes** (complaints) — fast, standard baseline for text classification
- **Decision Tree** (fake identity) — interpretable rules you can literally
  read out to a non-technical panel
- **Rule + statistics** (threat monitoring) — no ML needed; a z-score spike
  detector is exactly what real SOC/monitoring dashboards use