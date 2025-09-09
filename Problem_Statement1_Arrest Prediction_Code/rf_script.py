# rf_pca_script.py

import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "rf_model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--pca_variance", type=float, default=None)  # New argument for PCA
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    args = parser.parse_args()

    # Load data
    train = pd.read_csv(os.path.join(args.train, "train.csv"))
    validation = pd.read_csv(os.path.join(args.validation, "test.csv"))

    X_train = train.drop("Arrest", axis=1)
    y_train = train["Arrest"]
    X_val = validation.drop("Arrest", axis=1)
    y_val = validation["Arrest"]

    # Build Pipeline
    steps = [('scaler', StandardScaler())]

    if args.pca_variance:
        steps.append(('pca', PCA(n_components=args.pca_variance, random_state=42)))

    steps.append(('rf', RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )))

    pipeline = Pipeline(steps)

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_prob))

    # Save the whole pipeline
    joblib.dump(pipeline, os.path.join(args.model_dir, "rf_model.joblib"))
