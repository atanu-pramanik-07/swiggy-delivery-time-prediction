import json
import dagshub
import mlflow
import pytest
from mlflow import MlflowClient

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


model_name = load_model_information("run_information.json")["model_name"]


def test_load_model_from_registry():
    client = MlflowClient()

    staging_versions = client.get_latest_versions(name=model_name, stages=["Staging"])
    prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])

    if staging_versions:
        stage = "Staging"
        latest_version = staging_versions[0].version
    elif prod_versions:
        stage = "Production"
        latest_version = prod_versions[0].version
    else:
        pytest.fail(f"No model found in Staging or Production for '{model_name}'")

    model_path = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_path)

    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model v{latest_version} ({stage}) loaded successfully")