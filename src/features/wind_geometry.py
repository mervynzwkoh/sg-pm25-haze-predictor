"""Meteorological wind vector decomposition and transboundary advection geometry."""
import math
from typing import Tuple, Dict
from src.config import SG_CENTER, INDONESIA_UPSTREAM_STATIONS


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial bearing from point 1 to point 2 in degrees (0° - 360°)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def decompose_wind(speed: float, direction_deg: float) -> Tuple[float, float]:
    """Decompose wind speed (m/s or knots) and meteorological direction (deg from where wind blows)

    into zonal (u, eastward) and meridional (v, northward) components.
    """
    rad = math.radians(direction_deg)
    # Wind blowing *from* direction_deg
    u = -speed * math.sin(rad)  # Eastward wind component
    v = -speed * math.cos(rad)  # Northward wind component
    return u, v


def calculate_advection_alignment(wind_direction_deg: float, bearing_to_sg_deg: float) -> float:
    """Calculate cosine alignment between where the wind is blowing TO and the bearing towards Singapore.

    Returns value between -1.0 (opposing) and +1.0 (exact direct advection toward SG).
    """
    # Wind blows towards (wind_direction + 180) % 360
    blow_to_deg = (wind_direction_deg + 180.0) % 360.0
    angle_diff = abs(blow_to_deg - bearing_to_sg_deg)
    # Account for circle wrap-around
    if angle_diff > 180.0:
        angle_diff = 360.0 - angle_diff
    alignment = math.cos(math.radians(angle_diff))
    return max(0.0, alignment)  # Positive alignment only (0 if blowing away from SG)


def compute_transboundary_haze_risk(
    upstream_readings: Dict[str, float],
    wind_speed: float,
    wind_direction_deg: float
) -> Dict[str, float]:
    """Calculate transboundary advection risk scores from Indonesian fire clusters to Singapore.

    Returns individual station advection scores and an aggregated national transboundary risk index.
    """
    scores = {}
    weighted_total = 0.0
    total_weight = 0.0

    for station_name, info in INDONESIA_UPSTREAM_STATIONS.items():
        pm25 = upstream_readings.get(station_name, 50.0)
        bearing = info["bearing_to_sg"]
        dist = info["distance_km"]

        # Alignment towards Singapore
        alignment = calculate_advection_alignment(wind_direction_deg, bearing)

        # Advection intensity decays with distance and scales with wind speed
        decay_factor = 1.0 / (dist / 100.0)  # e.g. 35km Batam has higher immediate impact than 900km Kalimantan
        adv_score = pm25 * alignment * (wind_speed / 5.0) * decay_factor

        scores[f"adv_risk_{station_name.lower().replace(' ', '_')}"] = adv_score

        # Weight by station distance
        w = 1.0 / dist
        weighted_total += adv_score * w
        total_weight += w

    composite_score = (weighted_total / total_weight) if total_weight > 0 else 0.0
    scores["transboundary_haze_index"] = composite_score
    return scores
