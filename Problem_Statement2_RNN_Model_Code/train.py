# train.py - for AWS SageMaker RNN training from S3

import argparse
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.decomposition import PCA

def load_data(file_path, target_crime=25, window_size=12):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Date'].dt.floor('h')

    hourly = df.groupby(['Hour', 'Primary Type']).size().unstack(fill_value=0)
    hourly['total'] = hourly.sum(axis=1)
    hourly = hourly.sort_index()

    if target_crime not in hourly.columns:
        hourly[target_crime] = 0

    X_seq, y_seq = [], []
    features = list(hourly.columns)

    for i in range(len(hourly) - window_size - 1):
        x_window = hourly.iloc[i:i+window_size][features].values
        y_label = 1 if hourly.iloc[i+window_size][target_crime] > 0 else 0
        X_seq.append(x_window)
        y_seq.append(y_label)
    X = np.array(X_seq)
    y = np.array(y_seq)

    # --- PCA ---
    n_features = X.shape[2]
    X_flat = X.reshape(-1, n_features)  
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X_flat)  
    X = X_reduced.reshape(-1, window_size, 20)

    return X, y


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/opt/ml/input/data/training/df_clean.csv')
    parser.add_argument('--model_dir', type=str, required=True, help='S3 directory for saving the model')
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    args = parser.parse_args()

    print("Loading data from:", args.data_path)
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))

    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, class_weight=class_weights)

    model.save(os.path.join(args.output_dir, "crime_theft_rnn.h5"))
    print("Model saved to:", args.output_dir)


if __name__ == '__main__':
    main()
