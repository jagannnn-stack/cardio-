# cardio_project.py
# Cardiovascular Disease Prediction – Full Pipeline
# Author: Jagan :)
# Usage: python cardio_project.py

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------
# 1) Load Data
# -----------------------------
CSV_NAME_CANDIDATES = [
    "cardio_train.csv",
    "cardio_train (1).csv",
    "cardio_train (1).CSV",
]

csv_path = None
for name in CSV_NAME_CANDIDATES:
    if os.path.exists(name):
        csv_path = name
        break

if csv_path is None:
    raise FileNotFoundError(
        "CSV file not found. Keep 'cardio_train.csv' (semicolon-separated) in the same folder."
    )

df = pd.read_csv(csv_path, sep=";")
print("✅ Loaded:", csv_path)
print("Shape of dataset:", df.shape)
print(df.head(), "\n")

# -----------------------------
# 2) Basic Info + Missing
# -----------------------------
print("Dataset Info:")
print(df.info(), "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

# -----------------------------
# 3) Basic Cleaning / Sanity Filters
# (These ranges are widely used for this dataset to remove obvious outliers)
# -----------------------------
df["age_years"] = (df["age"] / 365).round().astype(int)
df["bmi"] = df["weight"] / (np.power(df["height"] / 100, 2))
df["ap_diff"] = df["ap_hi"] - df["ap_lo"]
df["ap_ratio"] = df["ap_hi"] / (df["ap_lo"].replace(0, np.nan))

# Filter unrealistic values
before = df.shape[0]
df = df[
    (df["height"].between(120, 220)) &
    (df["weight"].between(30, 200)) &
    (df["ap_hi"].between(80, 240)) &
    (df["ap_lo"].between(40, 200)) &
    (df["ap_hi"] >= df["ap_lo"]) &
    (df["age_years"].between(18, 100))
].copy()
after = df.shape[0]
print(f"🧹 Cleaned rows: removed {before - after}, remaining {after}\n")

# -----------------------------
# 4) EDA – Save simple plots
# -----------------------------
os.makedirs("plots", exist_ok=True)

# Target distribution
plt.figure(figsize=(5,4))
sns.countplot(x="cardio", data=df)
plt.title("Heart Disease (cardio) Distribution")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

# Age distribution
plt.figure(figsize=(7,4))
sns.histplot(df["age_years"], bins=30, kde=True)
plt.title("Age (years) Distribution")
plt.tight_layout()
plt.savefig("plots/age_distribution.png")
plt.close()

# BMI distribution
plt.figure(figsize=(7,4))
sns.histplot(df["bmi"], bins=30, kde=True)
plt.title("BMI Distribution")
plt.tight_layout()
plt.savefig("plots/bmi_distribution.png")
plt.close()

# Correlation heatmap (numeric only)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(11,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("📊 EDA plots saved inside the 'plots' folder.\n")

# -----------------------------
# 5) Train/Test Split
# -----------------------------
TARGET = "cardio"
X = df.drop(columns=[TARGET, "age"])  # we keep 'age_years' instead of raw 'age' in days
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Split:", X_train.shape, X_test.shape, "\n")

# -----------------------------
# 6) Build Models (use scaling where it helps)
# - Trees/Forest don't need scaling
# - LR/KNN/SVM benefit from scaling
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

# -----------------------------
# 7) Train, Evaluate, Compare
# -----------------------------
results = []
os.makedirs("plots/confusion_matrices", exist_ok=True)
os.makedirs("plots/roc_curves", exist_ok=True)
for name, model in models.items():
    print(f"🔧 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Some models may not support predict_proba; handle gracefully
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        # For models without predict_proba, approximate with decision_function if available
        try:
            scores = model.decision_function(X_test)
            # Convert to [0,1] via min-max for ROC-AUC calculation
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            roc = roc_auc_score(y_test, scores_norm)
        except Exception:
            roc = np.nan

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results.append([name, acc, prec, rec, f1, roc])

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrices/cm_{name.replace(' ', '_')}.png")
    plt.close()

# Results table
res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
res_df_sorted = res_df.sort_values(by=["F1","Accuracy"], ascending=False).reset_index(drop=True)

print("\n🏁 Model Comparison:")
print(res_df_sorted.to_string(index=False), "\n")

# Save results
res_df_sorted.to_csv("model_results.csv", index=False)
print("📄 Saved metrics table -> model_results.csv")

# -----------------------------
# 8) Pick Best Model, Save
# -----------------------------
best_model_name = res_df_sorted.iloc[0]["Model"]
best_model = models[best_model_name]
# Refit on full training set (already fitted, but safe)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "heart_model.pkl")
print(f"💾 Saved best model ({best_model_name}) -> heart_model.pkl\n")

# -----------------------------
# 9) Quick Demo Prediction (using one test row)
# -----------------------------
sample = X_test.iloc[[0]]
true_label = y_test.iloc[0]
pred = best_model.predict(sample)[0]
print("Demo Prediction on one sample:")
print("True label:", int(true_label), " Predicted:", int(pred))
print("\nAll set! Check the 'plots' folder for visuals and 'model_results.csv' for the comparison table.")
