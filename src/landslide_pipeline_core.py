# ============================================================
# Landslide Susceptibility Project
# Core ML Pipeline (Spatial CV + Stacking + Evaluation)
# ============================================================

import matplotlib
matplotlib.use("Agg")

import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Directory configuration
# ============================================================

BASE_DIR = Path("E:/landslide_project")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

MAPS_DIR = BASE_DIR / "maps"
BOUNDARY_DIR = MAPS_DIR / "boundary"

for d in [
    METRICS_DIR, PLOTS_DIR, SUMMARIES_DIR,
    MAPS_DIR, BOUNDARY_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

print("Directory structure ready.")

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(RAW_DATA_DIR / "tran_test_26_9.csv")
print("Dataset shape:", df.shape)

# ============================================================
# Spatial blocks
# ============================================================

BLOCK_SIZE = 1000  # meters

df["spatial_block_x"] = (df["X"] // BLOCK_SIZE).astype(int)
df["spatial_block_y"] = (df["Y"] // BLOCK_SIZE).astype(int)
df["spatial_block_id"] = (
    df["spatial_block_x"].astype(str) + "_" +
    df["spatial_block_y"].astype(str)
)

print("Unique spatial blocks:", df["spatial_block_id"].nunique())

# ============================================================
# Spatial train-test split
# ============================================================

from sklearn.model_selection import GroupShuffleSplit

TARGET = "Hazard"
EXCLUDE = [
    "Hazard", "X", "Y",
    "spatial_block_x", "spatial_block_y", "spatial_block_id"
]

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["spatial_block_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ============================================================
# Models & metrics
# ============================================================

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    roc_curve, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# ============================================================
# Base models (7)
# ============================================================

base_models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True))
    ])
}

if HAS_XGB:
    base_models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )

if HAS_LGBM:
    base_models["LightGBM"] = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, random_state=42
    )

if HAS_CAT:
    base_models["CatBoost"] = CatBoostClassifier(
        iterations=300, learning_rate=0.05,
        depth=6, verbose=False, random_state=42
    )

print("Models:", list(base_models.keys()))

# ============================================================
# Spatial Cross-Validation
# ============================================================

gkf = GroupKFold(n_splits=5)
cv_results = {}

for name, model in base_models.items():
    print(f"\n===== Spatial CV: {name} =====")
    aucs, accs = [], []

    for fold, (tr, val) in enumerate(gkf.split(X_train, y_train, groups_train)):
        cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        cal.fit(X_train.iloc[tr], y_train.iloc[tr])

        prob = cal.predict_proba(X_train.iloc[val])[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y_train.iloc[val], prob)
        acc = accuracy_score(y_train.iloc[val], pred)

        aucs.append(auc)
        accs.append(acc)

        print(f" Fold {fold+1}: AUC={auc:.3f}, ACC={acc:.3f}")

    cv_results[name] = {
        "mean_auc": np.mean(aucs),
        "mean_acc": np.mean(accs)
    }

# ---- PRINT + SAVE MEAN CV RESULTS ----
print("\n===== MEAN SPATIAL CV PERFORMANCE =====")
for k, v in cv_results.items():
    print(f"{k:20s} | Mean AUC={v['mean_auc']:.4f} | Mean ACC={v['mean_acc']:.4f}")

cv_summary = pd.DataFrame(cv_results).T.sort_values("mean_auc", ascending=False)
cv_summary.to_csv(SUMMARIES_DIR / "spatial_cv_model_ranking.csv")
cv_summary.to_csv(METRICS_DIR / "spatial_cv_metrics.csv")

# ============================================================
# Stacking (Top-3 models)
# ============================================================

top_models = cv_summary.head(3).index.tolist()
print("\nTop-3 models:", top_models)

stack_train = np.zeros((len(X_train), len(top_models)))
stack_test = np.zeros((len(X_test), len(top_models)))

for i, name in enumerate(top_models):
    model = base_models[name]
    oof = np.zeros(len(X_train))
    test_preds = []

    for tr, val in gkf.split(X_train, y_train, groups_train):
        cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        cal.fit(X_train.iloc[tr], y_train.iloc[tr])

        oof[val] = cal.predict_proba(X_train.iloc[val])[:, 1]
        test_preds.append(cal.predict_proba(X_test)[:, 1])

    stack_train[:, i] = oof
    stack_test[:, i] = np.mean(test_preds, axis=0)

# ---------------------------
# Meta learner (SAFE)
# ---------------------------
if HAS_XGB:
    meta_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
else:
    meta_model = LogisticRegression(max_iter=1000)

meta_model.fit(stack_train, y_train)

stack_prob = meta_model.predict_proba(stack_test)[:, 1]
stack_pred = (stack_prob >= 0.5).astype(int)

stack_auc = roc_auc_score(y_test, stack_prob)
stack_acc = accuracy_score(y_test, stack_pred)

print("\nSTACKED MODEL → AUC:", stack_auc, "ACC:", stack_acc)

# ============================================================
# Best single model (test)
# ============================================================

best_name = cv_summary.index[0]
best_model = base_models[best_name]

cal_best = CalibratedClassifierCV(best_model, cv=3, method="sigmoid")
cal_best.fit(X_train, y_train)

best_prob = cal_best.predict_proba(X_test)[:, 1]
best_pred = (best_prob >= 0.5).astype(int)

best_auc = roc_auc_score(y_test, best_prob)
best_acc = accuracy_score(y_test, best_pred)

print(f"\nBEST SINGLE ({best_name}) → AUC={best_auc:.4f}, ACC={best_acc:.4f}")

# ============================================================
# ROC & Confusion Matrices
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# ROC
fpr_s, tpr_s, _ = roc_curve(y_test, stack_prob)
fpr_b, tpr_b, _ = roc_curve(y_test, best_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_s, tpr_s, label="Stacked", linewidth=2)
plt.plot(fpr_b, tpr_b, label=best_name, linewidth=2)
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve Comparison")
plt.savefig(PLOTS_DIR / "roc_stacked_vs_best.png", dpi=300)
plt.close()

# Confusion matrices
def save_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(PLOTS_DIR / fname, dpi=300)
    plt.close()

save_cm(y_test, best_pred, f"CM – {best_name}", "cm_best_single.png")
save_cm(y_test, stack_pred, "CM – Stacked", "cm_stacked.png")

# ============================================================
# Bootstrap AUC significance test
# ============================================================

N_BOOT = 1000
rng = np.random.default_rng(42)
diffs = []

for _ in range(N_BOOT):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test.values[idx])) < 2:
        continue
    diffs.append(
        roc_auc_score(y_test.values[idx], stack_prob[idx]) -
        roc_auc_score(y_test.values[idx], best_prob[idx])
    )

diffs = np.array(diffs)

bootstrap_df = pd.DataFrame({
    "mean_auc_diff": [diffs.mean()],
    "ci_2.5": [np.percentile(diffs, 2.5)],
    "ci_97.5": [np.percentile(diffs, 97.5)],
    "p_value": [np.mean(diffs <= 0)]
})

bootstrap_df.to_csv(
    SUMMARIES_DIR / "bootstrap_auc_significance.csv",
    index=False
)

print("\n===== CORE PIPELINE COMPLETED SUCCESSFULLY =====")
