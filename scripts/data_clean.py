from typing import List

import numpy as np
import pandas as pd

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


def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(str.lower, axis=1).rename(
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
            # "time_taken(min)": "time_taken",
        },
        axis=1,
    )


def time_of_day(ser: pd.Series) -> pd.Series:
    return pd.cut(
        ser,
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=["after_midnight", "morning", "afternoon", "evening", "night"],
    ).astype("object")


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    # Identify minor riders (age < 18) and invalid 6-star rated drivers
    minor_index = data.loc[pd.to_numeric(data["age"], errors="coerce") < 18].index
    six_star_index = data.loc[
        pd.to_numeric(data["ratings"], errors="coerce") > 5
    ].index

    return (
        data.drop(columns="id")
        .drop(index=minor_index)  # Drop minor riders (age < 18)
        .drop(index=six_star_index)  # Drop 6-star rated drivers (ratings > 5)
        .replace("NaN ", np.nan)  # Convert string "NaN " to actual np.nan
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
                x["order_date"].dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
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
            # # Target column modifications
            # time_taken=lambda x: (
            #     x["time_taken"]
            #     .str.replace("(min) ", "", regex=False)
            #     .astype(int)
            # ),
        )
        .drop(columns=["order_time", "order_picked_time"])
    )


def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]
    return data.assign(
        **{
            col: np.where(data[col] < threshold, np.nan, data[col].values)
            for col in location_columns
        }
    )


def calculate_haversine_distance(data: pd.DataFrame) -> pd.DataFrame:
    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]
    lat1 = data[location_columns[0]]
    lon1 = data[location_columns[1]]
    lat2 = data[location_columns[2]]
    lon2 = data[location_columns[3]]

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


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        distance_type=pd.cut(
            data["distance"],
            bins=[0, 5, 10, 15, 25],
            right=False,
            labels=["short", "medium", "long", "very_long"],
        )
    )


def drop_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return data.drop(columns=columns)


def perform_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    clean_data = (
        data.pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=COLUMNS_TO_DROP)
    )

    return clean_data.dropna()