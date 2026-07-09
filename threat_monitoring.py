"""
FOCUS AREA 5: Digital Threat Monitoring
------------------------------------------
Simulates a real-time monitoring feed (e.g. scraped public posts, forum
threads, or complaint inflow) and flags emerging threat spikes - useful for
early warning on things like a new scam campaign, a riot-inciting hashtag
trending, or a sudden wave of similar complaints.

Approach: keyword/severity scoring + a rolling anomaly detector (z-score on
threat volume per hour) so the dashboard can alert when activity spikes
beyond normal baseline - no heavy model needed, which makes it easy to
explain to the police panel.
"""

import numpy as np
import pandas as pd

THREAT_KEYWORDS = {
    "high": ["bomb", "attack", "kill", "riot", "weapon", "explosive"],
    "medium": ["scam", "fraud", "hack", "phishing", "blackmail", "leak"],
    "low": ["suspicious", "fake", "spam", "unverified"],
}
SEVERITY_WEIGHT = {"high": 5, "medium": 3, "low": 1}


def score_post(text: str) -> int:
    text = text.lower()
    score = 0
    for level, words in THREAT_KEYWORDS.items():
        for w in words:
            if w in text:
                score += SEVERITY_WEIGHT[level]
    return score


def simulate_feed(hours=24, base_posts_per_hour=20, seed=1):
    """Simulates an incoming stream of posts/complaints over N hours,
    with an injected 'spike' event to prove the anomaly detector works."""
    rng = np.random.default_rng(seed)
    sample_texts = [
        "Just saw a suspicious link asking for OTP, be careful",
        "Great weather today, going for a walk",
        "This investment app looks like a scam, lost money",
        "Received threatening blackmail message online",
        "New phishing site impersonating the bank, spread awareness",
        "Someone is hacking accounts in our college group",
        "Normal day at work, nothing much happening",
        "Fake news spreading fast about riot in the city",
    ]

    records = []
    for hour in range(hours):
        n_posts = rng.poisson(base_posts_per_hour)
        if hour == 16:  # inject an anomaly spike at hour 16
            n_posts += 60
        for _ in range(n_posts):
            text = rng.choice(sample_texts)
            records.append({"hour": hour, "text": text, "threat_score": score_post(text)})
    return pd.DataFrame(records)


def detect_hourly_anomalies(df, z_threshold=2.0):
    hourly = df.groupby("hour")["threat_score"].sum().reindex(range(df["hour"].max() + 1), fill_value=0)
    mean, std = hourly.mean(), hourly.std()
    z_scores = (hourly - mean) / std
    alerts = hourly[z_scores > z_threshold]
    return hourly, alerts


if __name__ == "__main__":
    feed = simulate_feed()
    hourly, alerts = detect_hourly_anomalies(feed)

    print("=== Digital Threat Monitoring: Hourly Threat Volume ===")
    print(hourly)

    print("\n=== ALERTS: Hours with abnormal threat activity ===")
    if len(alerts) == 0:
        print("No anomalies detected.")
    else:
        for hour, score in alerts.items():
            print(f"Hour {hour}: threat_score={score} -> ALERT: possible emerging threat / coordinated activity")