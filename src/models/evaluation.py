import json
import logging
from pathlib import Path
from typing import Any, Tuple
import warnings
warnings.filterwarnings('ignore')
import dagshub
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET: str = "time_taken"
REPO_OWNER: str = "atanu-pramanik-07"
REPO_NAME: str = "swiggy-delivery-time-prediction"
MLFLOW_TRACKING_URI: str = (
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)
EXPERIMENT_NAME: str = "DVC Pipeline"
MODEL_NAME: str = "delivery_time_pred_model"


def setup_mlflow() -> None:
    """Initialize DagsHub and configure the MLflow tracking URI and experiment."""
    logger.info("Initializing DagsHub and MLflow experiment '%s'", EXPERIMENT_NAME)
    try:
        dagshub.init(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            mlflow=True,
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("MLflow tracking configured at %s", MLFLOW_TRACKING_URI)
    except Exception:
        logger.exception("Failed to initialize DagsHub/MLflow tracking")
        raise


def load_data(data_path: Path) -> pd.DataFrame:
    """Load CSV data from the given file path."""
    logger.info("Loading data from %s", data_path)
    try:
        df = pd.read_csv(data_path)
        logger.info("Loaded dataset with shape %s", df.shape)
        return df
    except FileNotFoundError:
        logger.exception("Data file not found at %s", data_path)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Data file at %s is empty", data_path)
        raise
    except Exception:
        logger.exception("Unexpected error while loading data from %s", data_path)
        raise


def make_X_and_y(
    data: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target series y."""
    logger.info("Splitting features and target column '%s'", target_column)
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        logger.info("Split complete. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y
    except KeyError:
        logger.exception("Target column '%s' not found in DataFrame", target_column)
        raise
    except Exception:
        logger.exception("Unexpected error while splitting X and y")
        raise


def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk using joblib."""
    logger.info("Loading model from %s", model_path)
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except FileNotFoundError:
        logger.exception("Model file not found at %s", model_path)
        raise
    except Exception:
        logger.exception("Unexpected error while loading model from %s", model_path)
        raise


def save_model_info(
    save_json_path: Path, run_id: str, artifact_path: str, model_name: str
) -> None:
    """Save MLflow run and artifact metadata to a JSON file."""
    logger.info("Saving model run information to %s", save_json_path)
    try:
        info_dict = {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "model_name": model_name,
        }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=4)
        logger.info("Model run information saved successfully to %s", save_json_path)
    except OSError:
        logger.exception("Failed to write model info JSON to %s", save_json_path)
        raise
    except Exception:
        logger.exception(
            "Unexpected error while saving model info to %s", save_json_path
        )
        raise


def main() -> None:
    """Entry point for evaluating the trained model and logging results to MLflow."""
    try:
        setup_mlflow()

        # Set file paths
        root_path = Path(__file__).parent.parent.parent
        train_data_path = root_path / "data" / "processed" / "train_trans.csv"
        test_data_path = root_path / "data" / "processed" / "test_trans.csv"
        model_path = root_path / "models" / "model.joblib"

        # Load train and test data
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        # Split train and test data into X and y
        X_train, y_train = make_X_and_y(train_data, TARGET)
        X_test, y_test = make_X_and_y(test_data, TARGET)

        # Load the trained model
        model = load_model(model_path)

        # Generate predictions
        logger.info("Generating predictions for train and test sets")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate MAE and R2 metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        logger.info(
            "Metrics - Train MAE: %.4f, Test MAE: %.4f, Train R2: %.4f, Test R2: %.4f",
            train_mae,
            test_mae,
            train_r2,
            test_r2,
        )

        # Calculate cross-validation scores
        logger.info("Running 5-fold cross-validation")
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mean_cv_score = -float(cv_scores.mean())
        logger.info("Cross-validation complete. Mean CV MAE: %.4f", mean_cv_score)

        # Log run details, metrics, datasets, and artifacts to MLflow
        logger.info("Logging run to MLflow")
        with mlflow.start_run() as run:
            mlflow.set_tag("model", "Food Delivery Time Regressor")

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("mean_cv_score", mean_cv_score)

            # Log individual CV fold scores
            mlflow.log_metrics(
                {f"CV {num}": float(score) for num, score in enumerate(-cv_scores)}
            )

            # Log input datasets
            train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
            test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
            mlflow.log_input(dataset=train_data_input, context="training")
            mlflow.log_input(dataset=test_data_input, context="validation")

            # Infer model signature
            sample_X = X_train.sample(20, random_state=42)
            model_signature = mlflow.models.infer_signature(
                model_input=sample_X,
                model_output=model.predict(sample_X),
            )

            # Log the final model
            mlflow.sklearn.log_model(
                model,
                name=MODEL_NAME,
                signature=model_signature,
                serialization_format="cloudpickle",
            )

            # Log supporting artifacts
            mlflow.log_artifact(
                str(root_path / "models" / "stacking_regressor.joblib")
            )
            mlflow.log_artifact(
                str(root_path / "models" / "power_transformer.joblib")
            )
            mlflow.log_artifact(
                str(root_path / "models" / "preprocessor.joblib")
            )

            artifact_uri = mlflow.get_artifact_uri()

        run_id = run.info.run_id
        logger.info("MLflow run completed successfully (run_id=%s)", run_id)

        # Save run metadata to JSON
        save_json_path = root_path / "run_information.json"
        save_model_info(
            save_json_path=save_json_path,
            run_id=run_id,
            artifact_path=artifact_uri,
            model_name=MODEL_NAME,
        )

        logger.info("Model evaluation pipeline completed successfully")
    except Exception:
        logger.critical("Model evaluation pipeline terminated with errors")
        raise


if __name__ == "__main__":
    main()