# mlp_script.py

import argparse
import os
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "mlp_model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker arguments
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    # MLP Hyperparameters
    parser.add_argument("--hidden_layer_sizes", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=300)

    # PCA option
    parser.add_argument("--pca_variance", type=float, default=0.0, help="Set variance to apply PCA. 0 means no PCA.")

    args = parser.parse_args()

    # Load data
    train = pd.read_csv(os.path.join(args.train, "train.csv"))
    val = pd.read_csv(os.path.join(args.validation, "test.csv"))

    X_train = train.drop("Arrest", axis=1)
    y_train = train["Arrest"]
    X_val = val.drop("Arrest", axis=1)
    y_val = val["Arrest"]

    # Apply PCA if requested
    if args.pca_variance > 0:
        print(f"Applying PCA with {args.pca_variance*100:.2f}% variance...")
        pca = PCA(n_components=args.pca_variance, svd_solver='full')
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    else:
        print("Skipping PCA...")

    # Define and train the MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(args.hidden_layer_sizes,),
        max_iter=args.max_iter,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_scores = model.predict_proba(X_val)[:, 1]

    print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_scores))

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "mlp_model.joblib"))
