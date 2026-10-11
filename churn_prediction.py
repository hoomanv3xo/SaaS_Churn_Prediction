"""
SaaS Churn Prediction
Generates a synthetic subscriber dataset, trains three models,
compares them, and saves the best one with joblib.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

RNG = np.random.default_rng(42)


# ---------- 1. Synthetic data ----------
def make_data(n=5000):
    df = pd.DataFrame({
        "tenure_months": RNG.integers(1, 61, n),
        "plan": RNG.choice(["Basic", "Pro", "Enterprise"], n, p=[0.5, 0.35, 0.15]),
        "monthly_price": 0.0,
        "logins_per_month": RNG.poisson(12, n),
        "support_tickets": RNG.poisson(1.5, n),
        "payment_failures": RNG.poisson(0.3, n),
        "seats": RNG.integers(1, 30, n),
    })
    df["monthly_price"] = df["plan"].map({"Basic": 15, "Pro": 40, "Enterprise": 120})

    # Churn probability driven by realistic factors
    z = (-1.0
         - 0.04 * df["tenure_months"]
         - 0.08 * df["logins_per_month"]
         + 0.35 * df["support_tickets"]
         + 0.9 * df["payment_failures"]
         + 0.5 * (df["plan"] == "Basic")
         - 0.02 * df["seats"]
         + 1.2)
    p = 1 / (1 + np.exp(-z))
    df["churned"] = (RNG.random(n) < p).astype(int)
    return df


# ---------- 2. Prepare ----------
df = make_data()
df.to_csv("saas_subscribers.csv", index=False)
print(f"Rows: {len(df)} | Churn rate: {df['churned'].mean():.1%}")

X = pd.get_dummies(df.drop(columns="churned"), columns=["plan"], drop_first=True)
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3. Models ----------
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="logloss", random_state=42),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    fitted[name] = model

results_df = pd.DataFrame(results).set_index("Model").round(3)
print("\nModel comparison:\n", results_df)
results_df.to_csv("model_comparison.csv")

# ---------- 4. Save best model ----------
best_name = results_df["ROC-AUC"].idxmax()
print(f"\nBest model by ROC-AUC: {best_name}")
joblib.dump({"model": fitted[best_name], "columns": list(X.columns)},
            "best_churn_model.joblib")
print(confusion_matrix(y_test, fitted[best_name].predict(X_test)))

# ---------- 5. Feature importance chart ----------
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 4), title="Random Forest feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved: saas_subscribers.csv, model_comparison.csv, "
      "best_churn_model.joblib, feature_importance.png")
