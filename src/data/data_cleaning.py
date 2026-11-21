import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

COLUMNS_TO_DROP: List[str] = [
    "rider_id",
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
    "order_date",
    "order_time_hour",
    "order_day",
    "city_name",
    "order_day_of_week",
    "order_month",
]

LOCATION_COLUMNS: List[str] = [
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
]


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


def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase and rename key fields."""
    logger.info("Standardizing column names")
    try:
        return (
            data.rename(str.lower, axis=1)
            .rename(
                {
                    "delivery_person_id": "rider_id",
                    "delivery_person_age": "age",
                    "delivery_person_ratings": "ratings",
                    "delivery_location_latitude": "delivery_latitude",
                    "delivery_location_longitude": "delivery_longitude",
                    "time_orderd": "order_time",
                    "time_order_picked": "order_picked_time",
                    "weatherconditions": "weather",
                    "road_traffic_density": "traffic",
                    "city": "city_type",
                    "time_taken(min)": "time_taken",
                },
                axis=1,
            )
        )
    except Exception:
        logger.exception("Failed to rename columns")
        raise


def time_of_day(ser: pd.Series) -> pd.Series:
    """Bin hour values into part-of-day categories."""
    try:
        return pd.cut(
            ser,
            bins=[0, 6, 12, 17, 20, 24],
            right=True,
            labels=["after_midnight", "morning", "afternoon", "evening", "night"],
        ).astype("object")
    except Exception:
        logger.exception("Failed to categorize time_of_day")
        raise


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw delivery records and engineer date/time features."""
    logger.info("Starting core data cleaning")
    try:
        # Identify minor riders (age < 18) and invalid 6-star rated drivers
        minor_index = data.loc[
            pd.to_numeric(data["age"], errors="coerce") < 18
        ].index
        six_star_index = data.loc[
            pd.to_numeric(data["ratings"], errors="coerce") > 5
        ].index

        logger.info(
            "Dropping %d minor rider rows and %d invalid (>5 star) rating rows",
            len(minor_index),
            len(six_star_index),
        )

        cleaned_df = (
            data
            .drop(columns="id")
            .drop(index=minor_index)
            .drop(index=six_star_index)
            .replace("NaN ", np.nan)
            .assign(
                # Clean rider_id and extract city_name
                rider_id=lambda x: x["rider_id"].str.strip(),
                city_name=lambda x: x["rider_id"].str.split("RES").str.get(0),
                # Convert numeric columns from object to float
                age=lambda x: x["age"].astype(float),
                ratings=lambda x: x["ratings"].astype(float),
                # Fix negative sign errors in location columns
                restaurant_latitude=lambda x: x["restaurant_latitude"].abs(),
                restaurant_longitude=lambda x: x["restaurant_longitude"].abs(),
                delivery_latitude=lambda x: x["delivery_latitude"].abs(),
                delivery_longitude=lambda x: x["delivery_longitude"].abs(),
                # Order date to datetime and date feature extraction
                order_date=lambda x: pd.to_datetime(x["order_date"], dayfirst=True),
                order_day=lambda x: x["order_date"].dt.day,
                order_month=lambda x: x["order_date"].dt.month,
                order_day_of_week=lambda x: x["order_date"].dt.day_name().str.lower(),
                is_weekend=lambda x: (
                    x["order_date"]
                    .dt.day_name()
                    .isin(["Saturday", "Sunday"])
                    .astype(int)
                ),
                # Time-based columns
                order_time=lambda x: pd.to_datetime(x["order_time"], format="mixed"),
                order_picked_time=lambda x: pd.to_datetime(
                    x["order_picked_time"], format="mixed"
                ),
                # Time taken to pick up order (handles midnight crossings via .dt.seconds)
                pickup_time_minutes=lambda x: (
                    (x["order_picked_time"] - x["order_time"]).dt.seconds / 60
                ),
                # Hour and part of day when order was placed
                order_time_hour=lambda x: x["order_time"].dt.hour,
                order_time_of_day=lambda x: x["order_time_hour"].pipe(time_of_day),
                # Categorical columns
                weather=lambda x: (
                    x["weather"]
                    .str.replace("conditions ", "", regex=False)
                    .str.lower()
                    .replace("nan", np.nan)
                ),
                traffic=lambda x: x["traffic"].str.rstrip().str.lower(),
                type_of_order=lambda x: x["type_of_order"].str.rstrip().str.lower(),
                type_of_vehicle=lambda x: x["type_of_vehicle"].str.rstrip().str.lower(),
                festival=lambda x: x["festival"].str.rstrip().str.lower(),
                city_type=lambda x: x["city_type"].str.rstrip().str.lower(),
                # Multiple deliveries column
                multiple_deliveries=lambda x: x["multiple_deliveries"].astype(float),
                # Target column modifications
                time_taken=lambda x: (
                    x["time_taken"]
                    .str.replace("(min) ", "", regex=False)
                    .astype(int)
                ),
            )
            .drop(columns=["order_time", "order_picked_time"])
        )

        logger.info("Core data cleaning complete. Shape: %s", cleaned_df.shape)
        return cleaned_df
    except KeyError:
        logger.exception("Missing expected column during data cleaning")
        raise
    except ValueError:
        logger.exception("Data type conversion error during data cleaning")
        raise
    except Exception:
        logger.exception("Unexpected error during data cleaning")
        raise


def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Replace latitude/longitude values below the threshold with NaN."""
    logger.info("Cleaning lat/long values below threshold %.2f", threshold)
    try:
        return data.assign(
            **{
                col: np.where(data[col] < threshold, np.nan, data[col].values)
                for col in LOCATION_COLUMNS
            }
        )
    except KeyError:
        logger.exception("Missing location column in clean_lat_long")
        raise
    except Exception:
        logger.exception("Unexpected error in clean_lat_long")
        raise


def calculate_haversine_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Haversine distance (in km) between restaurant and delivery locations."""
    logger.info("Calculating Haversine distance")
    try:
        lat1 = data[LOCATION_COLUMNS[0]]
        lon1 = data[LOCATION_COLUMNS[1]]
        lat2 = data[LOCATION_COLUMNS[2]]
        lon2 = data[LOCATION_COLUMNS[3]]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c

        return data.assign(distance=distance)
    except KeyError:
        logger.exception("Missing location column in calculate_haversine_distance")
        raise
    except Exception:
        logger.exception("Unexpected error in calculate_haversine_distance")
        raise


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    """Categorize distance into bins."""
    logger.info("Creating distance_type categories")
    try:
        return data.assign(
            distance_type=pd.cut(
                data["distance"],
                bins=[0, 5, 10, 15, 25],
                right=False,
                labels=["short", "medium", "long", "very_long"],
            )
        )
    except KeyError:
        logger.exception("Column 'distance' not found in create_distance_type")
        raise
    except Exception:
        logger.exception("Unexpected error in create_distance_type")
        raise


def drop_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop the specified columns from the DataFrame."""
    logger.info("Dropping %d columns: %s", len(columns), columns)
    try:
        return data.drop(columns=columns)
    except KeyError:
        logger.exception("One or more columns to drop were not found: %s", columns)
        raise
    except Exception:
        logger.exception("Unexpected error in drop_columns")
        raise


def perform_data_cleaning(data: pd.DataFrame, save_data_path: Path) -> pd.DataFrame:
    """Run the full data cleaning pipeline and save the result to CSV."""
    logger.info("Starting data cleaning pipeline")
    try:
        clean_data = (
            data
            .pipe(change_column_names)
            .pipe(data_cleaning)
            .pipe(clean_lat_long)
            .pipe(calculate_haversine_distance)
            .pipe(create_distance_type)
            .pipe(drop_columns, columns=COLUMNS_TO_DROP)
        )

        clean_data.to_csv(save_data_path, index= False)
        logger.info(
            "Cleaned data saved successfully to %s (final shape: %s)",
            save_data_path,
            clean_data.shape,
        )
        return clean_data
    except Exception:
        logger.exception("Data cleaning pipeline failed")
        raise


def main() -> None:
    """Entry point for loading, cleaning, and saving the dataset."""
    try:
        root_path = Path(__file__).parent.parent.parent
        cleaned_data_save_dir = root_path / "data" / "cleaned"
        cleaned_data_save_dir.mkdir(exist_ok=True, parents=True)

        cleaned_data_save_path = cleaned_data_save_dir / "swiggy_cleaned.csv"
        data_load_path = root_path / "data" / "raw" / "swiggy.csv"

        df = load_data(data_load_path)
        perform_data_cleaning(data=df, save_data_path=cleaned_data_save_path)
    except Exception:
        logger.critical("Pipeline execution terminated with errors")
        raise


if __name__ == "__main__":
    main()