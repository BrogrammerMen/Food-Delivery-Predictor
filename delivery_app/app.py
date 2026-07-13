"""
Varna Delivery Time Predictor
==============================
An interactive Streamlit demo that showcases a (placeholder) machine
learning model estimating food delivery times in Varna, Bulgaria.

This file is ONLY the web application layer. The trained model itself is
NOT included — see model_utils.py and README.md for exactly how to plug
your real `model.pkl` in.
"""

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from distance_utils import get_road_distance
from model_utils import (
    DeliveryInput,
    load_feature_importance,
    load_metrics,
    load_model,
    predict_delivery_time,
)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Varna Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
)

# --------------------------------------------------------------------------
# Static data
# --------------------------------------------------------------------------
VARNA_CENTER = (43.2141, 27.9147)

# Real-world coordinates of each restaurant in Varna.
RESTAURANTS = {
    "KFC": (43.2170784, 27.8971148),
    "McDonald's": (43.2130611, 27.9041275),
    "Domino's Pizza": (43.2077478, 27.9102520),
    "Гюрлата": (43.2301012, 27.8796576),
    "Le Chef": (43.2047846, 27.9188293),
    "Subway": (43.2102085, 27.9209451),
    "PizzaLab": (43.2169595, 27.8983326),
    "Hesburger": (43.2159810, 27.8956969),
    "Burger King": (43.2176775, 27.8982745),
    "Kebapa Papa": (43.2179717, 27.8979914),
}

# These option lists match exactly the categories the model was trained on
# (see model_utils.py: Traffic_Enc ordinal mapping, and the one-hot encoded
# weather / time-of-day / vehicle columns).
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
WEATHER_CONDITIONS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
TIME_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
VEHICLE_TYPES = ["Bike", "Car", "Scooter"]

RESTAURANT_ICONS = {
    "KFC": "drumstick-bite", "McDonald's": "burger", "Domino's Pizza": "pizza-slice",
    "Гюрлата": "utensils", "Le Chef": "hat-chef", "Subway": "bread-slice",
    "PizzaLab": "pizza-slice", "Hesburger": "burger", "Burger King": "burger",
    "Kebapa Papa": "utensils",
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_map(
    selected_restaurant: str,
    destination: tuple | None,
    route_coords: list[tuple] | None = None,
) -> folium.Map:
    fmap = folium.Map(location=VARNA_CENTER, zoom_start=13, tiles="CartoDB positron")

    for name, coords in RESTAURANTS.items():
        is_selected = name == selected_restaurant
        folium.Marker(
            location=coords,
            tooltip=f"{name}" + ("  (selected origin)" if is_selected else ""),
            icon=folium.Icon(
                color="green" if is_selected else "blue",
                icon=RESTAURANT_ICONS.get(name, "utensils"),
                prefix="fa",
            ),
        ).add_to(fmap)

    if destination is not None:
        folium.Marker(
            location=destination,
            tooltip="Delivery destination",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(fmap)

        origin_coords = RESTAURANTS[selected_restaurant]
        if route_coords:
            # Real road-network route geometry from the routing engine.
            folium.PolyLine(
                locations=route_coords,
                color="#2563EB",
                weight=5,
                opacity=0.8,
                tooltip="Road route",
            ).add_to(fmap)
        else:
            # Routing service unavailable — draw a dashed straight line so
            # it's visually clear this is only an approximation.
            folium.PolyLine(
                locations=[origin_coords, destination],
                color="#9CA3AF",
                weight=3,
                opacity=0.7,
                dash_array="8",
                tooltip="Estimated route (straight line)",
            ).add_to(fmap)

    return fmap


def reset_destination():
    st.session_state.destination = None
    st.session_state.prediction_result = None


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("destination", None)
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("last_inputs", None)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛵 Varna Delivery Time Predictor")
st.markdown(
    "A machine-learning demo that estimates how long a food delivery will "
    "take in **Varna, Bulgaria** — pick a restaurant, drop a destination "
    "pin, tune the delivery conditions, and get an instant prediction."
)

if load_model() is None:
    st.info(
        "🧪 **Demo Mode** — no trained model was found in `models/model.pkl`, "
        "so predictions below are produced by a simple placeholder formula "
        "instead of the real model. See README.md to plug in your trained "
        "model.",
        icon="🧪",
    )

# ==========================================================================
# Predict Delivery Time
# ==========================================================================
st.subheader("1. Choose the restaurant (origin)")
selected_restaurant = st.selectbox(
    "Restaurant",
    list(RESTAURANTS.keys()),
    label_visibility="collapsed",
)

st.subheader("2. Choose the delivery destination")
st.caption("Click anywhere on the map to drop the delivery destination pin.")

origin_coords = RESTAURANTS[selected_restaurant]
destination = st.session_state.destination

distance_km = None
distance_method = None
route_coords = None

if destination is not None:
    with st.spinner("Calculating road distance..."):
        distance_info = get_road_distance(origin_coords, destination)
    distance_km = distance_info["distance_km"]
    distance_method = distance_info["method"]
    route_coords = distance_info["route_coords"]

map_col, info_col = st.columns([3, 1])

with map_col:
    fmap = build_map(selected_restaurant, destination, route_coords)
    map_data = st_folium(
        fmap,
        height=480,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="delivery_map",
    )

    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        new_destination = (clicked["lat"], clicked["lng"])
        if new_destination != st.session_state.destination:
            st.session_state.destination = new_destination
            st.session_state.prediction_result = None
            st.rerun()

with info_col:
    st.markdown("**Origin**")
    st.write(f"📍 {selected_restaurant}")
    origin_lat, origin_lon = origin_coords
    st.caption(f"{origin_lat:.5f}, {origin_lon:.5f}")

    st.markdown("**Destination**")
    if destination:
        dlat, dlon = destination
        st.write("🏁 Selected")
        st.caption(f"{dlat:.5f}, {dlon:.5f}")

        if distance_method == "road_network":
            st.metric("🛣️ Road distance", f"{distance_km:.2f} km")
        else:
            st.metric("≈ Estimated distance", f"{distance_km:.2f} km")
            st.caption(
                "Routing service unreachable — estimated from the "
                "straight-line distance using a typical urban circuity "
                "factor instead of a real route."
            )
    else:
        st.write("_Not set yet — click the map._")

    st.button("🗑️ Clear destination", on_click=reset_destination, use_container_width=True)

st.divider()

st.subheader("3. Delivery conditions")
c1, c2, c3, c4 = st.columns(4)
traffic_level = c1.selectbox("Traffic level", TRAFFIC_LEVELS)
weather_condition = c2.selectbox("Weather condition", WEATHER_CONDITIONS)
time_of_day = c3.selectbox("Time of day", TIME_OF_DAY_OPTIONS)
preparation_time = c4.slider("Preparation time (minutes)", 2, 60, 15)

with st.expander("⚙️ Advanced settings"):
    st.caption(
        "These extra factors let the model take more context into account, "
        "even though our exploratory data analysis found they have a "
        "comparatively small effect on the final prediction."
    )
    a1, a2 = st.columns(2)
    courier_experience = a1.slider("Courier experience (years)", 0.0, 20.0, 2.0, step=0.5)
    vehicle_type = a2.selectbox("Delivery vehicle", VEHICLE_TYPES)

st.divider()

predict_clicked = st.button(
    "🔮 Predict Delivery Time",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.destination is None,
)
if st.session_state.destination is None:
    st.caption("Select a destination on the map above to enable prediction.")

if predict_clicked and st.session_state.destination is not None:
    delivery_input = DeliveryInput(
        distance_km=distance_km,
        preparation_time_min=preparation_time,
        traffic_level=traffic_level,
        weather_condition=weather_condition,
        time_of_day=time_of_day,
        courier_experience_years=courier_experience,
        vehicle_type=vehicle_type,
    )
    prediction_minutes, is_real_model = predict_delivery_time(delivery_input)
    st.session_state.prediction_result = (prediction_minutes, is_real_model)
    distance_label = (
        f"{distance_km:.2f} km (road route)"
        if distance_method == "road_network"
        else f"{distance_km:.2f} km (estimated)"
    )
    st.session_state.last_inputs = {
        "Restaurant (origin)": selected_restaurant,
        "Destination coordinates": f"{st.session_state.destination[0]:.5f}, {st.session_state.destination[1]:.5f}",
        "Distance": distance_label,
        "Traffic level": traffic_level,
        "Weather condition": weather_condition,
        "Time of day": time_of_day,
        "Preparation time": f"{preparation_time} min",
        "Courier experience": f"{courier_experience} yrs",
        "Vehicle type": vehicle_type,
    }

    # ---------------------- Results ----------------------
    if st.session_state.prediction_result:
        st.subheader("Result")
        prediction_minutes, is_real_model = st.session_state.prediction_result

        res_col, sum_col = st.columns([1, 2])
        with res_col:
            st.metric("⏱️ Estimated delivery time", f"{prediction_minutes:.0f} min")
            if not is_real_model:
                st.caption("⚠️ Produced by the Demo Mode placeholder formula, not a trained model.")

        with sum_col:
            st.markdown("**Inputs used for this prediction**")
            summary_df = pd.DataFrame(
                st.session_state.last_inputs.items(), columns=["Parameter", "Value"]
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

