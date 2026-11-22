import json
from pathlib import Path

import dagshub
import joblib
import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

dagshub.init(
    repo_owner="atanu-pramanik-07",
    repo_name="swiggy-delivery-time-prediction",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/atanu-pramanik-07/swiggy-delivery-time-prediction.mlflow"
)


def load_model_information(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transformer(transformer_path):
    return joblib.load(transformer_path)


def get_active_stage(model_name: str) -> str:
    client = MlflowClient()
    if client.get_latest_versions(name=model_name, stages=["Staging"]):
        return "Staging"
    if client.get_latest_versions(name=model_name, stages=["Production"]):
        return "Production"
    raise ValueError(f"No model found in Staging or Production for '{model_name}'")


# Set model name and resolve active stage
model_name = load_model_information("run_information.json")["model_name"]
stage = get_active_stage(model_name)

# Load the latest model from model registry
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Set paths and load preprocessor
root_path = Path(__file__).parent.parent
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# Build the model pipeline
model_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", model),
    ]
)

test_data_path = root_path / "data" / "interim" / "test.csv"


@pytest.mark.parametrize(
    argnames="model_pipe, test_data_path, threshold_error",
    argvalues=[(model_pipe, test_data_path, 5)],
)
def test_model_performance(model_pipe, test_data_path, threshold_error):
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)

    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    y_pred = model_pipe.predict(X)
    mean_error = mean_absolute_error(y, y_pred)

    assert (
        mean_error <= threshold_error
    ), f"The model does not pass the performance threshold of {threshold_error} minutes"
    print(f"The avg error is {mean_error:.4f} ({stage})")
    print(f"The {model_name} model passed the performance test")