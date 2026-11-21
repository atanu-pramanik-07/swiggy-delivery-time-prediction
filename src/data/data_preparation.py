import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "time_taken"


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


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test DataFrames."""
    logger.info(
        "Splitting data (test_size=%s, random_state=%s)", test_size, random_state
    )
    try:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        logger.info(
            "Data split complete. Train shape: %s, Test shape: %s",
            train_data.shape,
            test_data.shape,
        )
        return train_data, test_data
    except ValueError:
        logger.exception(
            "Invalid split parameters (test_size=%s, random_state=%s)",
            test_size,
            random_state,
        )
        raise
    except Exception:
        logger.exception("Unexpected error while splitting data")
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


def main() -> None:
    """Entry point for loading cleaned data, splitting it, and saving train/test sets."""
    try:
        # Set file paths
        root_path = Path(__file__).parent.parent.parent
        data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
        save_data_dir = root_path / "data" / "interim"
        save_data_dir.mkdir(exist_ok=True, parents=True)

        train_filename = "train.csv"
        test_filename = "test.csv"
        save_train_path = save_data_dir / train_filename
        save_test_path = save_data_dir / test_filename
        params_file_path = root_path / "params.yaml"

        # Load the cleaned data
        df = load_data(data_path)

        # Read the parameters
        try:
            parameters = read_params(params_file_path)["Data_Preparation"]
            test_size = parameters["test_size"]
            random_state = parameters["random_state"]
        except KeyError:
            logger.exception(
                "Missing expected key in %s (expected 'Data_Preparation', 'test_size', 'random_state')",
                params_file_path,
            )
            raise

        # Split into train and test data
        train_data, test_data = split_data(
            df, test_size=test_size, random_state=random_state
        )

        # Save the train and test data
        data_subsets = [train_data, test_data]
        data_paths = [save_train_path, save_test_path]
        filename_list = [train_filename, test_filename]

        for filename, path, data in zip(filename_list, data_paths, data_subsets):
            logger.info("Saving subset '%s'", filename)
            save_data(data=data, save_path=path)

        logger.info("Data preparation completed successfully")
    except Exception:
        logger.critical("Data preparation pipeline terminated with errors")
        raise


if __name__ == "__main__":
    main()