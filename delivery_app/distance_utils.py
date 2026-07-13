"""
distance_utils.py
===================
Estimates the real ROAD-NETWORK distance between two points (instead of a
straight, "as the crow flies" line) by querying a public OSRM routing
engine — the same technique used by most mapping apps: the actual
shortest path along the street network.

If the routing service can't be reached (offline, rate-limited, request
timeout, etc.) this falls back to a straight-line distance corrected by a
typical urban "circuity factor", so the app keeps working without a live
connection or an API key — it just labels the result as an estimate
rather than a real routed distance.

See README.md for notes on swapping in a different / self-hosted routing
provider for production use.
"""

from __future__ import annotations

import math
from typing import Optional

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Routing service configuration
# --------------------------------------------------------------------------
# The public OSRM demo server. Free, no API key required — but it's a
# shared, best-effort community resource, not meant for heavy production
# traffic. See README.md for self-hosting / commercial-provider notes.
OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving"
REQUEST_TIMEOUT_SECONDS = 6

# Typical urban "circuity factor" (real road distance / straight-line
# distance) used ONLY as a fallback when the routing service is
# unreachable. ~1.3 is a commonly cited average for city street networks.
FALLBACK_CIRCUITY_FACTOR = 1.3


def haversine_km(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Great-circle (straight-line) distance between two lat/lon points, in km."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0088  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data(show_spinner=False, ttl=3600)
def _query_osrm(origin: tuple[float, float], destination: tuple[float, float]) -> Optional[dict]:
    """
    Query OSRM for the driving route between two (lat, lon) points.

    Returns the raw `routes[0]` dict from OSRM's response (contains
    `distance` in meters and a GeoJSON `geometry`), or None if the request
    failed for any reason — network unavailable, timeout, rate limit, or
    no route found. Cached for an hour so re-running the Streamlit script
    (which happens on every widget interaction) doesn't re-fire the same
    request for an unchanged origin/destination pair.
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    url = f"{OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"][0]
    except (requests.RequestException, ValueError):
        # Covers connection errors, timeouts, bad JSON, non-2xx status, etc.
        pass
    return None


def get_road_distance(origin: tuple[float, float], destination: tuple[float, float]) -> dict:
    """
    Estimate the delivery distance between two points, preferring a real
    road-network route and falling back to an adjusted straight-line
    distance if routing isn't available.

    Returns:
        {
            "distance_km": float,
            "method": "road_network" | "estimated",
            "route_coords": list[tuple[float, float]] | None,  # (lat, lon) pairs for drawing on a map
        }
    """
    route = _query_osrm(origin, destination)
    if route is not None:
        distance_km = route["distance"] / 1000.0
        route_coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
        return {"distance_km": distance_km, "method": "road_network", "route_coords": route_coords}

    straight_km = haversine_km(origin, destination)
    return {
        "distance_km": straight_km * FALLBACK_CIRCUITY_FACTOR,
        "method": "estimated",
        "route_coords": None,
    }
