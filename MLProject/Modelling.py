# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import json
import argparse
import warnings

import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

warnings.filterwarnings("ignore")


# =========================================================
# ARGUMENT PARSER
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int, default=100)
parser.add_argument("--max_depth",         type=int, default=5)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()


# =========================================================
# SET MLFLOW TRACKING (DagsHub via env variable)
# =========================================================

MLFLOW_TRACKING_URI = "https://dagshub.com/evanursilviani2/Modelling_SML_EvaNurSilviani.mlflow"

os.environ["MLFLOW_TRACKING_URI"]      = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("Heart Disease CI")


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("heart_preprocessing.csv")


# =========================================================
# FEATURE DAN TARGET
# =========================================================

X = data.drop("target", axis=1)
y = data["target"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

param_grid = {
    "n_estimators":      [args.n_estimators, args.n_estimators * 2],
    "max_depth":         [args.max_depth, args.max_depth * 2],
    "min_samples_split": [args.min_samples_split, args.min_samples_split + 3]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

# =========================================================
# TRAIN
# =========================================================

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# =========================================================
# PREDICTION
# =========================================================

y_pred = best_model.predict(X_test)

# =========================================================
# METRICS
# =========================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# =========================================================
# MANUAL LOGGING — PARAMS
# =========================================================

mlflow.log_param("best_n_estimators",
                 grid_search.best_params_["n_estimators"])
mlflow.log_param("best_max_depth",
                 grid_search.best_params_["max_depth"])
mlflow.log_param("best_min_samples_split",
                 grid_search.best_params_["min_samples_split"])

# =========================================================
# MANUAL LOGGING — METRICS
# =========================================================

mlflow.log_metric("accuracy",  accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall",    recall)
mlflow.log_metric("f1_score",  f1)

# =========================================================
# ARTIFACTS
# =========================================================

os.makedirs("artifacts", exist_ok=True)

# 1. metric_info.json
metric_info = {
    "accuracy":  accuracy,
    "precision": precision,
    "recall":    recall,
    "f1_score":  f1
}
with open("artifacts/metric_info.json", "w") as f:
    json.dump(metric_info, f, indent=4)

# 2. classification report HTML
report = classification_report(y_test, y_pred)
html_content = f"""
<html>
<head><title>Classification Report</title></head>
<body>
<h1>Classification Report</h1>
<pre>{report}</pre>
</body>
</html>
"""
with open("artifacts/estimator.html", "w") as f:
    f.write(html_content)

# 3. confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("artifacts/training_confusion_matrix.png")
plt.close()

# log semua artifact
mlflow.log_artifact("artifacts/metric_info.json")
mlflow.log_artifact("artifacts/estimator.html")
mlflow.log_artifact("artifacts/training_confusion_matrix.png")

# =========================================================
# LOG MODEL
# =========================================================

input_example = X_train.iloc[:5]

mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    input_example=input_example
)

# =========================================================
# OUTPUT
# =========================================================

print("Training selesai!")
print("Best Parameters:", grid_search.best_params_)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")