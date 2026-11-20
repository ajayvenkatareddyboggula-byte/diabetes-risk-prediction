import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data load + same split
df = pd.read_csv("diabetes_clean.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Saved model + scaler load
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

X_test_s = scaler.transform(X_test)
X_test_s_df = pd.DataFrame(X_test_s, columns=X.columns)

# SHAP values
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test_s_df)

# Different shap versions different format istayi, rendu handle chestam
if isinstance(sv, list):
    sv_pos = sv[1]
    base_val = explainer.expected_value[1]
elif sv.ndim == 3:
    sv_pos = sv[:, :, 1]
    base_val = explainer.expected_value[1]
else:
    sv_pos = sv
    base_val = explainer.expected_value

# Graph 1: Summary plot (dots)
plt.figure()
shap.summary_plot(sv_pos, X_test_s_df, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()

# Graph 2: Bar plot (average importance)
plt.figure()
shap.summary_plot(sv_pos, X_test_s_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150)
plt.close()

# Graph 3: Oka patient explanation (test set lo first patient)
exp = shap.Explanation(values=sv_pos[0], base_values=base_val,
                       data=X_test.iloc[0].values, feature_names=list(X.columns))
plt.figure()
shap.plots.waterfall(exp, show=False)
plt.tight_layout()
plt.savefig("shap_patient.png", dpi=150)
plt.close()

# Text lo importance ranking
importance = pd.Series(np.abs(sv_pos).mean(axis=0), index=X.columns)
print("Feature importance (SHAP):")
print(importance.sort_values(ascending=False).round(4))
print("\nSaved: shap_summary.png, shap_bar.png, shap_patient.png")