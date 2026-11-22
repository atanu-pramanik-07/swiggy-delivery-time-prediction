import json
import logging
from pathlib import Path
from typing import Annotated, Any, Dict, Literal
from fastapi.staticfiles import StaticFiles
import dagshub
import joblib
import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sklearn import set_config
from sklearn.pipeline import Pipeline

from scripts.data_clean import perform_data_cleaning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set transformer outputs to pandas
set_config(transform_output="pandas")

REPO_OWNER: str = "atanu-pramanik-07"
REPO_NAME: str = "swiggy-delivery-time-prediction"
MLFLOW_TRACKING_URI: str = (
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)
MODEL_STAGE: str = "Production"
RUN_INFO_PATH: Path = Path("run_information.json")
PREPROCESSOR_PATH: Path = Path("models/preprocessor.joblib")

# Initialize DagsHub and MLflow
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class Data(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    ID: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            description="Unique order ID",
            examples=["0x4607"],
        ),
    ]
    Delivery_person_ID: Annotated[
        str,
        Field(
            ...,
            pattern=r"^[A-Z]+RES\d+DEL\d+$",
            description="Delivery person ID (e.g., INDORES13DEL02)",
            examples=["INDORES13DEL02"],
        ),
    ]
    Delivery_person_Age: Annotated[
        int,
        Field(
            ...,
            ge=18,
            le=65,
            description="Delivery partner age in years (valid dataset range: 20-50, must be >= 18)",
            examples=[34],
        ),
    ]
    Delivery_person_Ratings: Annotated[
        float,
        Field(
            ...,
            ge=1.0,
            le=5.0,
            description="Delivery partner rating between 1.0 and 5.0",
            examples=[4.5],
        ),
    ]
    Restaurant_latitude: Annotated[
        float,
        Field(
            ...,
            ge=6.0,
            le=38.0,
            description="Restaurant latitude in India (dataset valid range: 9.96 to 30.91)",
            examples=[22.745049],
        ),
    ]
    Restaurant_longitude: Annotated[
        float,
        Field(
            ...,
            ge=68.0,
            le=98.0,
            description="Restaurant longitude in India (dataset valid range: 72.77 to 88.43)",
            examples=[75.892471],
        ),
    ]
    Delivery_location_latitude: Annotated[
        float,
        Field(
            ...,
            ge=6.0,
            le=38.0,
            description="Delivery latitude in India (dataset valid range: 9.97 to 31.05)",
            examples=[22.765049],
        ),
    ]
    Delivery_location_longitude: Annotated[
        float,
        Field(
            ...,
            ge=68.0,
            le=98.0,
            description="Delivery longitude in India (dataset valid range: 72.78 to 88.56)",
            examples=[75.912471],
        ),
    ]
    Order_Date: Annotated[
        str,
        Field(
            ...,
            pattern=r"^\d{2}-\d{2}-\d{4}$",
            description="Order date in DD-MM-YYYY format",
            examples=["19-03-2022"],
        ),
    ]
    Time_Orderd: Annotated[
        str,
        Field(
            ...,
            pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
            description="Order placed time in HH:MM:SS (24-hour) format",
            examples=["11:30:00"],
        ),
    ]
    Time_Order_picked: Annotated[
        str,
        Field(
            ...,
            pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
            description="Order pickup time in HH:MM:SS (24-hour) format",
            examples=["11:45:00"],
        ),
    ]
    Weatherconditions: Annotated[
        Literal[
            "conditions Sunny",
            "conditions Stormy",
            "conditions Sandstorms",
            "conditions Cloudy",
            "conditions Fog",
            "conditions Windy",
        ],
        Field(..., description="Weather condition at order time"),
    ]
    Road_traffic_density: Annotated[
        Literal["Low", "Medium", "High", "Jam"],
        Field(..., description="Road traffic density level"),
    ]
    Vehicle_condition: Annotated[
        Literal[0, 1, 2, 3],
        Field(..., description="Vehicle condition rating (0 to 3)"),
    ]
    Type_of_order: Annotated[
        Literal["Snack", "Drinks", "Buffet", "Meal"],
        Field(..., description="Type of food order"),
    ]
    Type_of_vehicle: Annotated[
        Literal["motorcycle", "scooter", "electric_scooter", "bicycle"],
        Field(..., description="Delivery vehicle type"),
    ]
    multiple_deliveries: Annotated[
        Literal[0, 1, 2, 3],
        Field(..., description="Number of concurrent deliveries (0 to 3)"),
    ]
    Festival: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the order was placed during a festival"),
    ]
    City: Annotated[
        Literal["Urban", "Metropolitian", "Semi-Urban"],
        Field(
            ...,
            description="City category (uses dataset spelling 'Metropolitian')",
        ),
    ]


class PredicResponse(BaseModel):
    pred: Annotated[
        float,
        Field(..., description="Predicted delivery time in minutes"),
    ]


def load_model_information(file_path: Path) -> Dict[str, Any]:
    """Load run and model metadata from a JSON file."""
    logger.info("Loading model information from %s", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.exception("Run information file not found at %s", file_path)
        raise
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON file at %s", file_path)
        raise


def load_transformer(transformer_path: Path) -> Any:
    """Load a serialized preprocessor from disk using joblib."""
    logger.info("Loading preprocessor from %s", transformer_path)
    try:
        return joblib.load(transformer_path)
    except FileNotFoundError:
        logger.exception("Preprocessor file not found at %s", transformer_path)
        raise


# Columns to preprocess in data
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = [
    "weather",
    "type_of_order",
    "type_of_vehicle",
    "festival",
    "city_type",
    "is_weekend",
    "order_time_of_day",
]

ordinal_cat_cols = ["traffic", "distance_type"]

# Initialize MLflow client and load model artifacts
client = MlflowClient()
model_name = load_model_information(RUN_INFO_PATH)["model_name"]
# latest_model_ver = client.get_latest_versions(name=model_name, stages=[MODEL_STAGE])
# logger.info(
#     "Loaded latest '%s' versions for '%s': %s",
#     MODEL_STAGE,
#     model_name,
#     [v.version for v in latest_model_ver],
# )

model_path = f"models:/{model_name}/{MODEL_STAGE}"
logger.info("Loading model from MLflow registry: %s", model_path)
model = mlflow.sklearn.load_model(model_path)

preprocessor = load_transformer(PREPROCESSOR_PATH)

# Build the model pipeline
model_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", model),
    ]
)

# Create the FastAPI app
app = FastAPI(title="Swiggy Food Delivery Time Prediction API")


@app.get(path="/health")
def home() -> str:
    return "Welcome to the Swiggy Food Delivery Time Prediction App"


@app.post(path="/predict", response_model=PredicResponse)
def do_predictions(data: Data) -> PredicResponse:
    try:
        pred_data = pd.DataFrame(
            {
                "ID": data.ID,
                "Delivery_person_ID": data.Delivery_person_ID,
                "Delivery_person_Age": str(data.Delivery_person_Age),
                "Delivery_person_Ratings": str(data.Delivery_person_Ratings),
                "Restaurant_latitude": data.Restaurant_latitude,
                "Restaurant_longitude": data.Restaurant_longitude,
                "Delivery_location_latitude": data.Delivery_location_latitude,
                "Delivery_location_longitude": data.Delivery_location_longitude,
                "Order_Date": data.Order_Date,
                "Time_Orderd": data.Time_Orderd,
                "Time_Order_picked": data.Time_Order_picked,
                "Weatherconditions": data.Weatherconditions,
                "Road_traffic_density": data.Road_traffic_density,
                "Vehicle_condition": data.Vehicle_condition,
                "Type_of_order": data.Type_of_order,
                "Type_of_vehicle": data.Type_of_vehicle,
                "multiple_deliveries": str(data.multiple_deliveries),
                "Festival": data.Festival,
                "City": data.City,
            },
            index=[0],
        )

        # Clean the raw input data
        cleaned_data = perform_data_cleaning(pred_data)

        # Generate prediction
        prediction = float(model_pipe.predict(cleaned_data)[0])
        logger.info("Prediction generated successfully: %.2f min", prediction)

        return PredicResponse(pred=prediction)
    except Exception as exc:
        logger.exception("Prediction failed for order ID %s", data.ID)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

# # Keep this after all API routes so /predict is matched first
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
