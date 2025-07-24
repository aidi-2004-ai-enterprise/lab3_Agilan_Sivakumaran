import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import xgboost as xgb
import pandas as pd

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Path to saved model
MODEL_FILE = os.path.join(os.path.dirname(__file__), "data", "model.json")

app = FastAPI()


class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    male = "male"
    female = "female"


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island


# Loading Model
def load_model():
    logging.info("Loading model from %s", MODEL_FILE)
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.exception("Failed to load model.")
        raise e


model = load_model()


# Preprocessing Data
def preprocess_input(features: PenguinFeatures) -> pd.DataFrame:
    logger.info("Preprocessing feature input %s", features)
    try:
        input_dict = features.model_dump()
        X_input = pd.DataFrame([input_dict]) 
        X_input = pd.get_dummies(X_input, columns=["sex", "island"]) 
        expected_cols = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex_Female",
            "sex_Male",
            "island_Biscoe",
            "island_Dream",
            "island_Torgersen",
        ]
        X_input = X_input.reindex(columns=expected_cols, fill_value=0)
        X_input = X_input.astype(float)
        logger.info("Feature preprocessing completed successfully.")
        return X_input
    except Exception as e:
        logger.exception("Error during preprocessing.")
        raise e


# Creating the Routes
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: PenguinFeatures):
    logger.info("Received prediction request")
    try:
        X_input = preprocess_input(features)
        pred = model.predict(X_input.values)
        logger.info("Prediction successful. Result: %d", int(pred[0]))
        return {"prediction": int(pred[0])}
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction error")