# script.py

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def model_fn(model_dir):
    """Load the XGBoost model"""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
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
    parser.add_argument("--num_round", type=int, default=50)

    args = parser.parse_args()

    # Load training and validation data
    try:
        train_data = pd.read_csv(os.path.join(args.train, "train.csv"))
        val_data = pd.read_csv(os.path.join(args.validation, "test.csv"))
    except Exception as e:
        print("🚨 Error reading CSV files:", e)
        raise

    # Expect target to be last column
    try:
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1].astype(int)
        X_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1].astype(int)
    except Exception as e:
        print("🚨 Error splitting features and target:", e)
        raise

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dval = xgb.DMatrix(X_val.values, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "eval_metric": "logloss"
    }

    evals = [(dtrain, "train"), (dval, "validation")]

    try:
        model = xgb.train(params, dtrain, num_boost_round=args.num_round, evals=evals)
    except Exception as e:
        print("🚨 Error during model training:", e)
        raise

    # Make predictions
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.3).astype(int)

    try:
        print("\n✅ Classification Report:\n", classification_report(y_val, y_pred, digits=4))
        print("✅ Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("✅ ROC-AUC Score:", roc_auc_score(y_val, y_pred_proba))
    except Exception as e:
        print("🚨 Error calculating evaluation metrics:", e)

    # Save model
    model.save_model(os.path.join(args.model_dir, "xgboost-model"))
