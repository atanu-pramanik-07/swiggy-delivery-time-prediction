import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set transformer outputs to pandas
set_config(transform_output="pandas")

# Columns to preprocess
NUM_COLS: List[str] = [
    "age",
    "ratings",
    "pickup_time_minutes",
    "distance",
]

NOMINAL_CAT_COLS: List[str] = [
    "weather",
    "type_of_order",
    "type_of_vehicle",
    "festival",
    "city_type",
    "is_weekend",
    "order_time_of_day",
]

ORDINAL_CAT_COLS: List[str] = ["traffic", "distance_type"]

TARGET_COL: str = "time_taken"

# Order for ordinal encoding
TRAFFIC_ORDER: List[str] = ["low", "medium", "high", "jam"]
DISTANCE_TYPE_ORDER: List[str] = ["short", "medium", "long", "very_long"]


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV data from the given file path."""
    logger.info("Loading data from %s", path)
    try:
        df = pd.read_csv(path)
        logger.info("Loaded dataset with shape %s", df.shape)
        return df
    except FileNotFoundError:
        logger.exception("Data file not found at %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("Data file at %s is empty", path)
        raise
    except Exception:
        logger.exception("Unexpected error while loading data from %s", path)
        raise


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows containing missing values from the DataFrame."""
    logger.info("Dropping missing values from DataFrame with shape %s", data.shape)
    try:
        df_dropped = data.dropna()
        missing_values = int(df_dropped.isnull().sum().sum())
        rows_dropped = len(data) - len(df_dropped)
        logger.info(
            "Dropped %d rows. Remaining shape: %s (remaining missing values: %d)",
            rows_dropped,
            df_dropped.shape,
            missing_values,
        )
        return df_dropped
    except Exception:
        logger.exception("Failed to drop missing values")
        raise


def save_transformer(
    transformer: ColumnTransformer, save_dir: Path, transformer_name: str
) -> None:
    """Serialize and save the fitted transformer using joblib."""
    save_location = save_dir / transformer_name
    logger.info("Saving transformer to %s", save_location)
    try:
        joblib.dump(value=transformer, filename=save_location)
        logger.info("Transformer saved successfully to %s", save_location)
    except OSError:
        logger.exception("Failed to write transformer to %s", save_location)
        raise
    except Exception:
        logger.exception("Unexpected error while saving transformer to %s", save_location)
        raise


def train_preprocessor(
    preprocessor: ColumnTransformer, data: pd.DataFrame
) -> ColumnTransformer:
    """Fit the preprocessor on the training feature DataFrame."""
    logger.info("Fitting preprocessor on data with shape %s", data.shape)
    try:
        preprocessor.fit(data)
        logger.info("Preprocessor fitted successfully")
        return preprocessor
    except KeyError:
        logger.exception("Missing expected feature column while fitting preprocessor")
        raise
    except ValueError:
        logger.exception("Invalid data encountered while fitting preprocessor")
        raise
    except Exception:
        logger.exception("Unexpected error while fitting preprocessor")
        raise


def perform_transformations(
    preprocessor: ColumnTransformer, data: pd.DataFrame
) -> pd.DataFrame:
    """Transform the feature DataFrame using the fitted preprocessor."""
    logger.info("Transforming data with shape %s", data.shape)
    try:
        transformed_data = preprocessor.transform(data)
        logger.info(
            "Transformation complete. Output shape: %s", transformed_data.shape
        )
        return transformed_data
    except KeyError:
        logger.exception("Missing expected feature column during transformation")
        raise
    except ValueError:
        logger.exception("Invalid data encountered during transformation")
        raise
    except Exception:
        logger.exception("Unexpected error during transformation")
        raise


def save_data(data: pd.DataFrame, save_path: Path) -> None:
    """Save a DataFrame to CSV without the index."""
    logger.info("Saving data (shape %s) to %s", data.shape, save_path)
    try:
        data.to_csv(save_path, index=False)
        logger.info("Successfully saved data to %s", save_path)
    except OSError:
        logger.exception("Failed to write file to %s", save_path)
        raise
    except Exception:
        logger.exception("Unexpected error while saving data to %s", save_path)
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


def join_X_and_y(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Join transformed features X and target y on their index."""
    logger.info("Joining transformed features and target")
    try:
        joined_df = X.join(y, how="inner")
        logger.info("Join complete. Combined shape: %s", joined_df.shape)
        return joined_df
    except Exception:
        logger.exception("Unexpected error while joining X and y")
        raise


def main() -> None:
    """Entry point for preprocessing train/test data and saving artifacts."""
    try:
        # Set paths
        root_path = Path(__file__).parent.parent.parent
        train_data_path = root_path / "data" / "interim" / "train.csv"
        test_data_path = root_path / "data" / "interim" / "test.csv"

        save_data_dir = root_path / "data" / "processed"
        save_data_dir.mkdir(exist_ok=True, parents=True)

        train_trans_filename = "train_trans.csv"
        test_trans_filename = "test_trans.csv"
        save_train_trans_path = save_data_dir / train_trans_filename
        save_test_trans_path = save_data_dir / test_trans_filename

# Configure preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("scale", MinMaxScaler(), NUM_COLS),
                (
                    "nominal_encode",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    NOMINAL_CAT_COLS,
                ),
                (
                    "ordinal_encode",
                    OrdinalEncoder(
                        categories=[TRAFFIC_ORDER, DISTANCE_TYPE_ORDER],
                        encoded_missing_value=-999,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ORDINAL_CAT_COLS,
                ),
            ],
            remainder="passthrough",
            n_jobs=-1,
            verbose_feature_names_out=False,
        )

        # Load train and test data with missing values dropped
        train_df = drop_missing_values(load_data(path=train_data_path))
        test_df = drop_missing_values(load_data(path=test_data_path))

        # Split train and test data into X and y
        X_train, y_train = make_X_and_y(data=train_df, target_column=TARGET_COL)
        X_test, y_test = make_X_and_y(data=test_df, target_column=TARGET_COL)

        # Fit preprocessor on X_train
        train_preprocessor(preprocessor=preprocessor, data=X_train)

        # Transform train and test features
        X_train_trans = perform_transformations(
            preprocessor=preprocessor, data=X_train
        )
        X_test_trans = perform_transformations(
            preprocessor=preprocessor, data=X_test
        )

        # Join transformed X and y back together
        train_trans_df = join_X_and_y(X_train_trans, y_train)
        test_trans_df = join_X_and_y(X_test_trans, y_test)

        # Save transformed datasets
        data_subsets = [train_trans_df, test_trans_df]
        data_paths = [save_train_trans_path, save_test_trans_path]
        filename_list = [train_trans_filename, test_trans_filename]

        for filename, path, data in zip(filename_list, data_paths, data_subsets):
            logger.info("Saving transformed subset '%s'", filename)
            save_data(data=data, save_path=path)

        # Save the fitted preprocessor
        transformer_filename = "preprocessor.joblib"
        transformer_save_dir = root_path / "models"
        transformer_save_dir.mkdir(exist_ok=True, parents=True)

        save_transformer(
            transformer=preprocessor,
            save_dir=transformer_save_dir,
            transformer_name=transformer_filename,
        )

        logger.info("Data preprocessing completed successfully")
    except Exception:
        logger.critical("Data preprocessing pipeline terminated with errors")
        raise


if __name__ == "__main__":
    main()