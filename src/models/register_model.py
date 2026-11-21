import json
import logging
from pathlib import Path
from typing import Any, Dict

import dagshub
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_OWNER: str = "atanu-pramanik-07"
REPO_NAME: str = "swiggy-delivery-time-prediction"
MLFLOW_TRACKING_URI: str = (
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)
TARGET_STAGE: str = "Staging"


def setup_mlflow() -> None:
    """Initialize DagsHub and configure the MLflow tracking URI."""
    logger.info("Initializing DagsHub and MLflow tracking")
    try:
        dagshub.init(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            mlflow=True,
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
    except Exception:
        logger.exception("Failed to initialize DagsHub/MLflow tracking")
        raise


def load_model_information(file_path: Path) -> Dict[str, Any]:
    """Load run and model metadata from a JSON file."""
    logger.info("Loading model information from %s", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        logger.info("Loaded model information successfully")
        return run_info
    except FileNotFoundError:
        logger.exception("Run information file not found at %s", file_path)
        raise
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON file at %s", file_path)
        raise
    except Exception:
        logger.exception(
            "Unexpected error while loading model information from %s", file_path
        )
        raise


def main() -> None:
    """Entry point for registering the model in MLflow and transitioning it to Staging."""
    try:
        setup_mlflow()

        # Set file paths
        root_path = Path(__file__).parent.parent.parent
        run_info_path = root_path / "run_information.json"

        # Load run information
        run_info = load_model_information(run_info_path)

        try:
            run_id = run_info["run_id"]
            model_name = run_info["model_name"]
            model_registry_path = run_info["model_uri"]
        except KeyError:
            logger.exception(
                "Missing expected key in %s (expected 'run_id' and 'model_name')",
                run_info_path,
            )
            raise


        logger.info(
            "Registering model '%s' from URI '%s'", model_name, model_registry_path
        )
        try:
            model_version = mlflow.register_model(
                model_uri=model_registry_path,
                name=model_name,
            )
        except MlflowException:
            logger.exception(
                "MLflow error while registering model '%s' from '%s'",
                model_name,
                model_registry_path,
            )
            raise

        registered_model_version = model_version.version
        registered_model_name = model_version.name
        logger.info(
            "Registered model '%s' version %s",
            registered_model_name,
            registered_model_version,
        )

        # Transition the model version stage to Staging
        logger.info(
            "Transitioning model '%s' version %s to stage '%s'",
            registered_model_name,
            registered_model_version,
            TARGET_STAGE,
        )
        try:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=registered_model_name,
                version=registered_model_version,
                stage=TARGET_STAGE,
            )
        except MlflowException:
            logger.exception(
                "MLflow error while transitioning model '%s' version %s to '%s'",
                registered_model_name,
                registered_model_version,
                TARGET_STAGE,
            )
            raise

        logger.info(
            "Model '%s' version %s successfully transitioned to '%s'",
            registered_model_name,
            registered_model_version,
            TARGET_STAGE,
        )
    except Exception:
        logger.critical("Model registration pipeline terminated with errors")
        raise


if __name__ == "__main__":
    main()