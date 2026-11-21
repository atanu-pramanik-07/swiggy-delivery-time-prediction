from pathlib import Path

import pandas as pd
import requests

# Path for data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# Prediction endpoint
predict_url = "http://127.0.0.1:8000/predict"

# Load and filter to a valid row (removes literal "NaN ", minors, >5 ratings, bad coords)
df = pd.read_csv(data_path)
df = df.apply(
    lambda col: col.str.strip() if pd.api.types.is_string_dtype(col) else col
)
df = df.replace(["NaN", "conditions NaN"], pd.NA).dropna()

df = df[
    (pd.to_numeric(df["Delivery_person_Age"], errors="coerce") >= 18)
    & (pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce") <= 5.0)
    & (df["Restaurant_latitude"].between(6.0, 38.0))
    & (df["Restaurant_longitude"].between(68.0, 98.0))
    & (df["Delivery_location_latitude"].between(6.0, 38.0))
    & (df["Delivery_location_longitude"].between(68.0, 98.0))
]

# Sample one valid row for testing
sample_row = df.sample(1)
target_col = sample_row.columns[-1]
print("The target value is", sample_row[target_col].item().replace("(min) ", ""))

# Build payload matching FastAPI Pydantic types
data = sample_row.drop(columns=[target_col]).squeeze().to_dict()
data["Delivery_person_Age"] = int(data["Delivery_person_Age"])
data["Delivery_person_Ratings"] = float(data["Delivery_person_Ratings"])
data["Vehicle_condition"] = int(data["Vehicle_condition"])
data["multiple_deliveries"] = int(data["multiple_deliveries"])
print(data)

# Send request to API
response = requests.post(url=predict_url, json=data, timeout=30)
print("The status code for response is", response.status_code)

if response.status_code == 200:
    pred = response.json()["pred"]
    print(f"The prediction value by the API is {pred:.2f} min")
else:
    print("Error:", response.status_code, response.text)