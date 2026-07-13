"""
model_utils.py
================
All logic related to the machine learning model lives here: loading the
trained artifacts, preprocessing raw form input into the exact shape the
model expects, and producing a prediction.

>>> THIS FILE CONTAINS PLACEHOLDERS <<<
The actual trained model is NOT included in this project. Everywhere you
see a "PLACEHOLDER" comment, that is a spot you need to touch once you
plug in your real `model.pkl`. See README.md for the full integration
guide.

Until real artifacts are added to the `models/` folder, the app runs in
"Demo Mode": a simple, transparent formula stands in for the model so the
full user experience (map -> inputs -> prediction -> results) can still be
demonstrated end-to-end.

--------------------------------------------------------------------------
INPUT ENCODING — this matches a sample of the real training data:

    Distance_km, Preparation_Time_min, Courier_Experience_yrs, Traffic_Enc,
    Clear, Foggy, Rainy, Snowy, Windy,
    Afternoon, Evening, Morning, Night,
    Bike, Car, Scooter

  - Traffic_Enc is ORDINAL: Low=1, Medium=2, High=3
  - Weather, time of day, and vehicle type are ONE-HOT encoded: exactly one
    of the corresponding dummy columns is True/1 per row, the rest False/0.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Paths where the real, trained artifacts are expected to live.
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "Food_Delivery_Time_LineRegress.pkl")  # your trained estimator
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")   # PLACEHOLDER: optional extra scaler/transformer
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")  # PLACEHOLDER: optional

# --------------------------------------------------------------------------
# Encoding definitions — MUST match how the training data was encoded.
# PLACEHOLDER: if your actual training columns/order differ, edit these.
# --------------------------------------------------------------------------
TRAFFIC_ENCODING = {"Low": 1, "Medium": 2, "High": 3}          # ordinal
WEATHER_OPTIONS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]        # one-hot
TIME_OF_DAY_OPTIONS = ["Afternoon", "Evening", "Morning", "Night"]     # one-hot
VEHICLE_OPTIONS = ["Bike", "Car", "Scooter"]                            # one-hot

# The exact column order the model was trained on (this is the final,
# fully-encoded feature vector fed into `model.predict(...)`).
MODEL_FEATURE_COLUMNS = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Traffic_Enc",
    "Clear", "Foggy", "Rainy", "Snowy", "Windy",
    "Afternoon", "Evening", "Morning", "Night",
    "Bike", "Car", "Scooter",
]

# Example / demo feature importances shown until a real ranking is supplied.
# PLACEHOLDER: replace with the importances your trained model actually learned
# (e.g. from model.feature_importances_ or a permutation-importance study),
# or drop a `feature_importance.csv` file into `models/` (columns: feature,importance)
# and it will be picked up automatically instead of these example values.
# NOTE: these are grouped at the conceptual-feature level (not one per
# one-hot dummy column) since that reads more clearly in a chart.
EXAMPLE_FEATURE_IMPORTANCE = {
    "distance_km": 0.33,
    "preparation_time_min": 0.25,
    "traffic_level": 0.15,
    "weather_condition": 0.10,
    "time_of_day": 0.08,
    "vehicle_type": 0.05,
    "courier_experience_years": 0.04,
}

# Example / demo evaluation metrics shown until real ones are supplied.
# PLACEHOLDER: replace with your actual 5-fold cross-validation results.
EXAMPLE_METRICS = {
    "mae_minutes": 6.4,
    "r2_score": 0.81,
    "cv_folds": 5,
}


@dataclass
class DeliveryInput:
    """Container for every value collected from the UI."""
    distance_km: float
    preparation_time_min: float
    traffic_level: str          # one of TRAFFIC_ENCODING keys ("Low"/"Medium"/"High")
    weather_condition: str      # one of WEATHER_OPTIONS
    time_of_day: str            # one of TIME_OF_DAY_OPTIONS
    courier_experience_years: float
    vehicle_type: str           # one of VEHICLE_OPTIONS


# --------------------------------------------------------------------------
# Loading the trained artifacts
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained model from disk.

    PLACEHOLDER: this function assumes a scikit-learn-compatible estimator
    was saved with joblib/pickle as `models/model.pkl`. If you saved your
    model differently (e.g. a Keras model, a custom class, XGBoost's native
    format, etc.) replace the loading call below accordingly.

    Returns None if no model file is present yet, so the rest of the app
    can gracefully fall back to Demo Mode.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"Found a model file but couldn't load it: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """
    Load an OPTIONAL, separately-saved extra preprocessing step — for
    example a fitted `StandardScaler` applied to the numeric columns
    (`Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`)
    during training.

    PLACEHOLDER: the ordinal traffic encoding and the one-hot weather /
    time-of-day / vehicle encoding are already handled deterministically
    in `preprocess_input()` below, since they're fixed, known mappings.
    This hook is only for anything additional your training pipeline did
    on top of that (e.g. scaling). If nothing else was done, you can leave
    `models/preprocessor.pkl` absent and this is skipped entirely.
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        return None
    try:
        import joblib
        return joblib.load(PREPROCESSOR_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"Found a preprocessor file but couldn't load it: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    """Load feature importance from disk if provided, else fall back to the example values."""
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        try:
            df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
            return dict(zip(df["feature"], df["importance"]))
        except Exception:
            pass
    return EXAMPLE_FEATURE_IMPORTANCE


def load_metrics() -> dict:
    """
    PLACEHOLDER: hard-coded example metrics. Swap these for the real
    numbers you obtained during model evaluation (see README.md), or wire
    this function up to read them from a small metrics.json file the way
    `load_feature_importance` reads a CSV.
    """
    return EXAMPLE_METRICS


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess_input(data: DeliveryInput) -> pd.DataFrame:
    """
    Turn the raw values collected from the Streamlit form into the exact
    encoded, tabular shape the trained model expects — mirroring the
    training data:

        Distance_km, Preparation_Time_min, Courier_Experience_yrs,
        Traffic_Enc, Clear, Foggy, Rainy, Snowy, Windy,
        Afternoon, Evening, Morning, Night, Bike, Car, Scooter

    - Traffic level is ORDINAL-encoded (Low=1, Medium=2, High=3).
    - Weather, time of day, and vehicle type are ONE-HOT encoded.

    PLACEHOLDER: if your model's training pipeline additionally scaled the
    numeric columns (Distance_km / Preparation_Time_min /
    Courier_Experience_yrs), save that fitted scaler as
    `models/preprocessor.pkl` — it will be applied automatically below.
    """
    row = {
        "Distance_km": data.distance_km,
        "Preparation_Time_min": data.preparation_time_min,
        "Courier_Experience_yrs": data.courier_experience_years,
        "Traffic_Enc": TRAFFIC_ENCODING[data.traffic_level],
    }

    # One-hot encode weather / time of day / vehicle type.
    for option in WEATHER_OPTIONS:
        row[option] = (option == data.weather_condition)
    for option in TIME_OF_DAY_OPTIONS:
        row[option] = (option == data.time_of_day)
    for option in VEHICLE_OPTIONS:
        row[option] = (option == data.vehicle_type)

    df = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)

    preprocessor = load_preprocessor()
    if preprocessor is not None:
        # PLACEHOLDER: adjust this call if your preprocessor's API differs
        # (e.g. it might only touch a subset of columns, or need
        # .transform vs .fit_transform, etc.)
        return preprocessor.transform(df)

    return df


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _demo_mode_estimate(data: DeliveryInput) -> float:
    """
    A transparent, deterministic-ish stand-in for the real model so the
    app is fully demoable before a trained model is available.

    This is NOT a machine learning model — it is a simple weighted
    formula loosely inspired by typical patterns in food-delivery data
    (distance and prep time dominate; traffic/weather/rush-hour add
    friction; experience and vehicle type add small adjustments). It
    exists purely so reviewers can click "Predict" and see the full UX
    work end to end.
    """
    traffic_penalty = {"Low": 0, "Medium": 5, "High": 11}
    weather_penalty = {"Clear": 0, "Windy": 2, "Foggy": 5, "Rainy": 6, "Snowy": 9}
    time_of_day_penalty = {"Morning": 3, "Afternoon": 5, "Evening": 6, "Night": 0}
    vehicle_speed_factor = {"Bicycle": 1.6, "Bike": 1.6, "Scooter": 1.1, "Car": 0.9}

    base_travel_time = data.distance_km * 3.2 * vehicle_speed_factor.get(data.vehicle_type, 1.0)
    experience_bonus = max(0.0, 3.0 - data.courier_experience_years * 0.15)

    estimate = (
        data.preparation_time_min
        + base_travel_time
        + traffic_penalty.get(data.traffic_level, 5)
        + weather_penalty.get(data.weather_condition, 2)
        + time_of_day_penalty.get(data.time_of_day, 2)
        + experience_bonus
    )

    # tiny bit of jitter so repeated identical inputs don't look "too fake"
    random.seed(int(data.distance_km * 1000) + int(data.preparation_time_min))
    estimate += random.uniform(-1.5, 1.5)

    return max(5.0, estimate)


def predict_delivery_time(data: DeliveryInput) -> tuple[float, bool]:
    """
    Returns (predicted_minutes, is_real_model).

    PLACEHOLDER: once `models/model.pkl` exists, this automatically
    switches from the demo formula to `model.predict(...)`.
    """
    model = load_model()

    if model is not None:
        try:
            processed = preprocess_input(data)
            # PLACEHOLDER: adjust if your model needs a numpy array instead
            # of a DataFrame, or returns something other than a single float.
            prediction = model.predict(processed)
            value = float(np.ravel(prediction)[0])
            return value, True
        except Exception as exc:  # pragma: no cover - defensive
            st.error(
                "A trained model was found but prediction failed "
                f"({exc}). Falling back to Demo Mode for this request."
            )

    return _demo_mode_estimate(data), False
