import json
import dagshub
import mlflow
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


def promote_model():
    model_name = load_model_information("run_information.json")["model_name"]
    stage = "Staging"
    promotion_stage = "Production"

    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

    if not latest_versions:
        print(f"No model in '{stage}' stage for '{model_name}'. Skipping promotion.")
        return

    latest_model_version_staging = latest_versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_model_version_staging,
        stage=promotion_stage,
        archive_existing_versions=True,
    )
    print(
        f"Promoted '{model_name}' v{latest_model_version_staging} "
        f"from '{stage}' to '{promotion_stage}'."
    )


if __name__ == "__main__":
    promote_model()