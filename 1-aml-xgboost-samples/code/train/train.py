import os
import argparse
import gzip
import struct
import mlflow
import numpy as np

from azureml.core import Run
from azureml.core.model import Model

import xgboost as xgb
from sklearn.metrics import accuracy_score

def get_aml_workspace():
    run = Run.get_context()
    ws = run.experiment.workspace
    return ws

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_known_args()[0]

    return args

def load_dataset(dataset_path):
    def unpack_mnist_data(filename: str, label=False):
        with gzip.open(filename) as gz:
            struct.unpack('I', gz.read(4))
            n_items = struct.unpack('>I', gz.read(4))
            if not label:
                n_rows = struct.unpack('>I', gz.read(4))[0]
                n_cols = struct.unpack('>I', gz.read(4))[0]
                res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
                res = res.reshape(n_items[0], n_rows * n_cols) / 255.0
            else:
                res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
                res = res.reshape(-1)
        return res
    
    X_train = unpack_mnist_data(os.path.join(dataset_path, 'train-images.gz'), False)
    y_train = unpack_mnist_data(os.path.join(dataset_path, 'train-labels.gz'), True)
    X_test = unpack_mnist_data(os.path.join(dataset_path, 'test-images.gz'), False)
    y_test = unpack_mnist_data(os.path.join(dataset_path, 'test-labels.gz'), True)

    return X_train, y_train, X_test, y_test

def create_model():
    return xgb.XGBClassifier(use_label_encoder=False, max_depth=3, n_estimators=10)

def train_model(X, y, model_filename):    
    model = create_model()
    model.fit(X, y, eval_metric='mlogloss', verbose=True)
    model.save_model(model_filename)

def evaluate_model(X, y, model_filename):
    model = create_model()
    model.load_model(model_filename)
    preds = model.predict(X)
    accscore = accuracy_score(y, preds)

    mlflow.log_metric('accuracy', accscore)

def register_model(ws, model_filename):
    model = Model.register(
        workspace=ws,
        model_name=model_filename,
        model_path=model_filename
    )

def main():
    args = parse_arguments()

    ws = get_aml_workspace()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.start_run()

    X_train, y_train, X_test, y_test = load_dataset(args.data)

    model_filename = "mnist.xgb_model"
    train_model(X_train, y_train, model_filename)
    evaluate_model(X_test, y_test, model_filename)
    register_model(ws, model_filename)
    
if __name__ == "__main__":
    main()