# combined_xgb_script.py

import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker-specific arguments
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--eval_metric", type=str, default="logloss")
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--num_round", type=int, default=50)

    # PCA controls
    parser.add_argument("--apply_pca", type=int, default=0)  # 1 = apply PCA, 0 = don't
    parser.add_argument("--pca_variance", type=float, default=0.9)  # Variance to preserve if PCA is applied

    args = parser.parse_args()

    # Load training and validation data
    train_data = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_data = pd.read_csv(os.path.join(args.validation, "test.csv"))

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1].astype(int)
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1].astype(int)

    # Apply PCA if specified
    if args.apply_pca:
        print(f"\n✅ Applying PCA to retain {args.pca_variance * 100}% variance...")
        pca = PCA(n_components=args.pca_variance, svd_solver='full')
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    else:
        print("\n✅ PCA not applied. Proceeding with original features.")

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "objective": args.objective,
        "tree_method": args.tree_method,
    }

    evals = [(dtrain, "train"), (dval, "validation")]

    model = xgb.train(params, dtrain, num_boost_round=args.num_round, evals=evals)

    # Predict
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Evaluate
    print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_pred_proba))

    # Save model
    model.save_model(os.path.join(args.model_dir, "xgboost-model"))