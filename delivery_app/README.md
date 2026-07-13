# Varna Delivery Time Predictor 🛵

A Streamlit demo application that showcases a machine learning model
estimating food delivery times in **Varna, Bulgaria**. The user picks a
restaurant, drops a destination pin on an interactive map, tunes the
delivery conditions, and gets an instant predicted delivery time.

> **Scope of this repo:** only the *web application* is implemented here.
> The trained machine learning model itself is **not included** — this
> README explains exactly where and how to plug your `model.pkl` in.

---

## 1. Project structure

```
varna-delivery-app/
├── app.py                 # Streamlit UI — map, form, results, model info tab
├── distance_utils.py       # Road-network distance (routing + fallback estimate)
├── model_utils.py          # All model-related logic (loading / preprocessing / predicting)
├── requirements.txt
├── models/                                    # <-- put your trained artifacts here
│   ├── Food_Delivery_Time_LineRegress.pkl          # (you provide) the trained estimator
│   ├── preprocessor.pkl                            # (optional, you provide) fitted encoder/scaler
│   └── feature_importance.csv                      # (optional, you provide) real feature importances
└── README.md
```

## 2. Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Until a real model is added, the app runs in **Demo Mode**: predictions
come from a simple, transparent placeholder formula (see
`_demo_mode_estimate` in `model_utils.py`) instead of a trained model, so
the full user experience can be reviewed end-to-end before the model is
wired in. A banner at the top of the app tells you which mode is active.

---

## 3. How to plug in your trained model

All of the following lives in **`model_utils.py`**. Every spot you need to
touch is marked with a `PLACEHOLDER` comment in the code.

### Step 1 — Drop your model file in `models/Food_Delivery_Time_LineRegress.pkl`

The app expects a **scikit-learn-compatible estimator** (anything with a
`.predict()` method) saved with `joblib`:

```python
import joblib
joblib.dump(trained_model, "models/Food_Delivery_Time_LineRegress.pkl")
```

If you trained with something else (XGBoost's native booster, a Keras/
PyTorch model, etc.), open `load_model()` in `model_utils.py` and swap the
`joblib.load(...)` call for the appropriate loader. The expected filename
is set by the `MODEL_PATH` constant near the top of `model_utils.py` —
rename it there if your file is called something else.

As soon as `models/Food_Delivery_Time_LineRegress.pkl` exists, the app
automatically switches from Demo Mode to using your real model — no other
code changes required for this step alone.

### Step 2 — Match the preprocessing pipeline

This is the most important step: **the app must transform raw form input
into the exact same encoded shape your model was trained on.**

`preprocess_input()` in `model_utils.py` is already wired up to match a
sample of the real training data that was provided for this project. Each
prediction is turned into a single-row `pandas.DataFrame` with these 16
columns, in this exact order:

| Column                     | Type          | Encoding                                             | Source in the UI                         |
|------------------------------|---------------|--------------------------------------------------------|--------------------------------------------|
| `Distance_km`                 | float         | raw value                                                 | computed from map origin/destination        |
| `Preparation_Time_min`        | float         | raw value                                                 | "Preparation time" slider                     |
| `Courier_Experience_yrs`      | float         | raw value                                                 | Advanced settings slider                        |
| `Traffic_Enc`                  | int           | **ordinal**: Low=1, Medium=2, High=3                        | "Traffic level" dropdown                          |
| `Clear`/`Foggy`/`Rainy`/`Snowy`/`Windy` | bool | **one-hot** (exactly one `True`)                          | "Weather condition" dropdown                       |
| `Afternoon`/`Evening`/`Morning`/`Night` | bool | **one-hot** (exactly one `True`)                          | "Time of day" dropdown                               |
| `Bike`/`Car`/`Scooter`         | bool          | **one-hot** (exactly one `True`)                          | Advanced settings "Delivery vehicle" dropdown           |

If your real training data uses different column names, a different
column order, or different categories, edit the constants near the top of
`model_utils.py`: `MODEL_FEATURE_COLUMNS`, `TRAFFIC_ENCODING`,
`WEATHER_OPTIONS`, `TIME_OF_DAY_OPTIONS`, `VEHICLE_OPTIONS` — the
dropdown options shown in `app.py` are driven directly from these lists,
so updating them here keeps the whole app consistent automatically.

The ordinal traffic mapping and the one-hot encoding are applied
deterministically in code (no separate encoder object needed for that
part). If your training pipeline did something *additional* on top of
this — most commonly, scaling the three numeric columns with a
`StandardScaler`/`MinMaxScaler` — save that fitted object too:

```python
import joblib
joblib.dump(fitted_scaler, "models/preprocessor.pkl")
```

`preprocess_input()` will automatically detect and apply it via
`preprocessor.transform(df)` right after the encoded row is assembled.
Double-check the `.transform()` call matches your preprocessor's API
(e.g. whether it expects/returns a DataFrame vs. a plain numpy array).

If, instead, your saved `model.pkl` is a full `sklearn.Pipeline` that
already contains its own preprocessing step (so it can accept the encoded
row — or even fully raw values — directly), you don't need
`preprocessor.pkl` at all; just make sure `MODEL_FEATURE_COLUMNS` matches
what that pipeline expects as input.

### Step 3 — Update the "About the Model" metrics

Open `EXAMPLE_METRICS` near the top of `model_utils.py` and replace the
placeholder Mean Absolute Error / R² / fold-count with your actual 5-fold
cross-validation results:

```python
EXAMPLE_METRICS = {
    "mae_minutes": <your real MAE>,
    "r2_score": <your real R²>,
    "cv_folds": 5,
}
```

(Or wire `load_metrics()` up to read a small `metrics.json` file if you'd
rather not hard-code numbers.)

### Step 4 — Update the feature importance chart

Two options:

* Simplest: edit the `EXAMPLE_FEATURE_IMPORTANCE` dictionary in
  `model_utils.py` with your model's real importances.
* Or drop a CSV at `models/feature_importance.csv` with two columns,
  `feature,importance` — it's picked up automatically and takes priority
  over the hard-coded example values.

### Step 5 — Test it

```bash
streamlit run app.py
```

The "Demo Mode" banner should disappear, and predictions should now come
from `model.predict(...)` on your real model.

---

## 4. How road distance is calculated

`distance_utils.py` estimates the delivery distance between the selected
restaurant and the destination pin using the **actual road network**,
not a straight line:

1. **Primary — real routing.** It queries the public
   [OSRM](https://project-osrm.org/) demo routing server
   (`router.project-osrm.org`), which computes the real shortest driving
   route along the street network and returns both the route distance and
   its geometry. The app draws that real route as a solid blue line on
   the map, and feeds the routed distance (in km) into the model as
   `Distance_km`.
2. **Fallback — estimated distance.** If the routing service can't be
   reached (no internet connection, request timeout, rate limiting, or no
   route found), the app falls back to the straight-line (haversine)
   distance multiplied by a typical urban **circuity factor** (`1.3`,
   configurable via `FALLBACK_CIRCUITY_FACTOR`) — a standard technique for
   approximating road distance from straight-line distance when a real
   route isn't available. In this case the map shows a dashed grey line
   instead of a solid route, and the UI clearly labels the number as
   "≈ Estimated distance" rather than "🛣️ Road distance", both live and
   in the results summary table.

Notes for production use:

* The public OSRM demo server is a shared, best-effort community
  resource — fine for a demo/MVP, but not intended for heavy or
  commercial traffic. For production, either self-host OSRM (or another
  routing engine) or switch to a commercial routing API (Mapbox
  Directions, Google Directions, OpenRouteService, etc.) by editing
  `OSRM_BASE_URL` and the request/response handling in
  `_query_osrm()`/`get_road_distance()`.
* Results are cached per (origin, destination) pair for an hour
  (`st.cache_data(..., ttl=3600)`) so re-running the Streamlit script
  (which happens on every widget interaction) doesn't repeatedly hit the
  routing service for the same two points.
* The routing profile is currently fixed to `driving`, since that's the
  only profile guaranteed to be available on the public OSRM demo
  instance. If you self-host OSRM with `cycling`/`walking` profiles
  enabled, you could route by the selected vehicle type instead.

---

## 5. Notes on the data / restaurants

* Restaurant markers use real coordinates in Varna, Bulgaria for: KFC,
  McDonald's, Domino's Pizza, Гюрлата, Le Chef, Subway, PizzaLab,
  Hesburger, Burger King, and Kebapa Papa.
* The distance feature is the real **road-network driving distance**
  between origin and destination (via OSRM routing), falling back to an
  adjusted straight-line estimate if routing is unavailable — see
  section 4 above for details.
* The traffic / weather / time-of-day / vehicle category options match a
  sample of the real training data provided for this project: traffic is
  ordinal-encoded (`Low`/`Medium`/`High`), and weather
  (`Clear`/`Foggy`/`Rainy`/`Snowy`/`Windy`), time of day
  (`Morning`/`Afternoon`/`Evening`/`Night`), and vehicle type
  (`Bike`/`Car`/`Scooter`) are one-hot encoded. A "Time of day" dropdown
  was added to the delivery-conditions section for this reason — it
  wasn't in the original mockup, but the model was trained with it as a
  feature. If your real dataset differs, update the option lists in
  `app.py` (`TRAFFIC_LEVELS`, `WEATHER_CONDITIONS`, `TIME_OF_DAY_OPTIONS`,
  `VEHICLE_TYPES`) and the matching encodings in `model_utils.py`
  (`TRAFFIC_ENCODING`, `WEATHER_OPTIONS`, `TIME_OF_DAY_OPTIONS`,
  `VEHICLE_OPTIONS`, `MODEL_FEATURE_COLUMNS`).
