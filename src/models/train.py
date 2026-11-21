import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET: str = "time_taken"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load CSV data from the given file path."""
    logger.info("Loading training data from %s", data_path)
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


def read_params(file_path: Path) -> Dict[str, Any]:
    """Read parameters from a YAML configuration file."""
    logger.info("Reading parameters from %s", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            params_file = yaml.safe_load(f)
        if params_file is None:
            raise ValueError(f"YAML file at {file_path} is empty")
        logger.info("Parameters loaded successfully")
        return params_file
    except FileNotFoundError:
        logger.exception("Parameters file not found at %s", file_path)
        raise
    except yaml.YAMLError:
        logger.exception("Failed to parse YAML file at %s", file_path)
        raise
    except Exception:
        logger.exception("Unexpected error while reading parameters from %s", file_path)
        raise


def save_model(model: Any, save_dir: Path, model_name: str) -> None:
    """Serialize and save a trained model using joblib."""
    save_location = save_dir / model_name
    logger.info("Saving model to %s", save_location)
    try:
        joblib.dump(value=model, filename=save_location)
        logger.info("Model saved successfully to %s", save_location)
    except OSError:
        logger.exception("Failed to write model to %s", save_location)
        raise
    except Exception:
        logger.exception("Unexpected error while saving model to %s", save_location)
        raise


def save_transformer(transformer: Any, save_dir: Path, transformer_name: str) -> None:
    """Serialize and save a fitted transformer using joblib."""
    save_location = save_dir / transformer_name
    logger.info("Saving transformer to %s", save_location)
    try:
        joblib.dump(transformer, save_location)
        logger.info("Transformer saved successfully to %s", save_location)
    except OSError:
        logger.exception("Failed to write transformer to %s", save_location)
        raise
    except Exception:
        logger.exception(
            "Unexpected error while saving transformer to %s", save_location
        )
        raise


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit the model on the training features and target."""
    logger.info(
        "Starting model training on X_train shape %s and y_train shape %s",
        X_train.shape,
        y_train.shape,
    )
    try:
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return model
    except ValueError:
        logger.exception("Invalid data or hyperparameters encountered during training")
        raise
    except Exception:
        logger.exception("Unexpected error during model training")
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


def main() -> None:
    """Entry point for training the stacking model and saving artifacts."""
    try:
        # Set paths
        root_path = Path(__file__).parent.parent.parent
        data_path = root_path / "data" / "processed" / "train_trans.csv"
        params_file_path = root_path / "params.yaml"

        # Load the training data
        training_data = load_data(data_path)

        # Split the data into X and y
        X_train, y_train = make_X_and_y(training_data, TARGET)

        # Read model parameters
        try:
            model_params = read_params(params_file_path)["Train"]
            rf_params = model_params["Random_Forest"]
            lgbm_params = model_params["LightGBM"]
        except KeyError:
            logger.exception(
                "Missing expected key in %s (expected 'Train', 'Random_Forest', 'LightGBM')",
                params_file_path,
            )
            raise

        # Build base models, meta model, and target transformer
        logger.info("Initializing RandomForestRegressor with params: %s", rf_params)
        rf = RandomForestRegressor(**rf_params)

        logger.info("Initializing LGBMRegressor with params: %s", lgbm_params)
        lgbm = LGBMRegressor(**lgbm_params)

        lr = LinearRegression()
        power_transform = PowerTransformer()

        # Form the stacking regressor
        stacking_reg = StackingRegressor(
            estimators=[
                ("rf_model", rf),
                ("lgbm_model", lgbm),
            ],
            final_estimator=lr,
            cv=5,
            n_jobs=-1,
        )

        # Make the model wrapper
        model = TransformedTargetRegressor(
            regressor=stacking_reg,
            transformer=power_transform,
        )

        # Fit the model on training data
        train_model(model, X_train, y_train)

        # Directory to save models and transformer
        model_save_dir = root_path / "models"
        model_save_dir.mkdir(exist_ok=True, parents=True)

        # Extract the model and transformer from wrapper
        stacking_model = model.regressor_
        transformer = model.transformer_

        # Save the full wrapped model
        model_filename = "model.joblib"
        save_model(
            model=model,
            save_dir=model_save_dir,
            model_name=model_filename,
        )

        # Save the stacking model
        stacking_filename = "stacking_regressor.joblib"
        save_model(
            model=stacking_model,
            save_dir=model_save_dir,
            model_name=stacking_filename,
        )

        # Save the transformer
        transformer_filename = "power_transformer.joblib"
        save_transformer(
            transformer=transformer,
            save_dir=model_save_dir,
            transformer_name=transformer_filename,
        )

        logger.info("Model training pipeline completed successfully")
    except Exception:
        logger.critical("Model training pipeline terminated with errors")
        raise


if __name__ == "__main__":
    main()