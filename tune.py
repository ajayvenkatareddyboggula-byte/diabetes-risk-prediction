import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)

# Data load + split (same as before)
df = pd.read_csv("diabetes_clean.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Before tuning (default model)
base = RandomForestClassifier(n_estimators=200, random_state=42)
base.fit(X_train_s, y_train)
base_pred = base.predict(X_test_s)
print("BEFORE tuning -> Recall:", round(recall_score(y_test, base_pred), 3),
      "| F1:", round(f1_score(y_test, base_pred), 3))

# Tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
)
grid.fit(X_train_s, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV recall:", round(grid.best_score_, 3))

# After tuning
best = grid.best_estimator_
pred = best.predict(X_test_s)
proba = best.predict_proba(X_test_s)[:, 1]

print("\nAFTER tuning on test set:")
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1       :", round(f1_score(y_test, pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 3))
print("\n", classification_report(y_test, pred))

# Save for next steps (SHAP + Streamlit)
joblib.dump(best, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved model.pkl and scaler.pkl")