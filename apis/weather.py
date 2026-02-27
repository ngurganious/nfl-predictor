"""
Open-Meteo Weather Client
==========================
Completely free, no API key required.
https://open-meteo.com/en/docs

Provides game-day weather forecasts by stadium location:
  - Temperature (°F)
  - Wind speed (mph)
  - Precipitation probability (%)
  - Precipitation amount (mm)

IMPORTANT: Weather only applies to outdoor stadiums.
Dome/retractable-roof stadiums always return indoor defaults.

Usage:
    from apis.weather import WeatherClient
    client = WeatherClient()

    weather = client.get_game_weather("Arrowhead Stadium", "2024-12-01T18:00")
    # Returns: {"temp": 28, "wind": 14, "precip_prob": 60, "is_dome": 0}

    weather = client.get_weather_for_teams("KC", "BUF", "2024-12-01T18:00")
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from . import cache

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"

TTL_WEATHER_FORECAST = 60 * 60 * 3    # 3 hrs — forecasts change
TTL_WEATHER_PAST     = 60 * 60 * 24   # 24 hrs — historical is static

# ── Stadium database ─────────────────────────────────────────────────────────
# lat, lon, is_dome, surface
# is_dome: 1 = always dome/indoor, 0 = outdoor or retractable (weather matters)
STADIUMS: dict[str, dict] = {
    # AFC East
    "BUF": {"name": "Highmark Stadium",          "lat": 42.7738,  "lon": -78.7870,  "is_dome": 0, "surface": "grass"},
    "MIA": {"name": "Hard Rock Stadium",          "lat": 25.9580,  "lon": -80.2389,  "is_dome": 0, "surface": "grass"},
    "NE":  {"name": "Gillette Stadium",           "lat": 42.0909,  "lon": -71.2643,  "is_dome": 0, "surface": "grass"},
    "NYJ": {"name": "MetLife Stadium",            "lat": 40.8135,  "lon": -74.0744,  "is_dome": 0, "surface": "turf"},
    # AFC North
    "BAL": {"name": "M&T Bank Stadium",           "lat": 39.2780,  "lon": -76.6227,  "is_dome": 0, "surface": "grass"},
    "CIN": {"name": "Paycor Stadium",             "lat": 39.0954,  "lon": -84.5160,  "is_dome": 0, "surface": "turf"},
    "CLE": {"name": "Cleveland Browns Stadium",   "lat": 41.5061,  "lon": -81.6995,  "is_dome": 0, "surface": "grass"},
    "PIT": {"name": "Acrisure Stadium",           "lat": 40.4468,  "lon": -80.0158,  "is_dome": 0, "surface": "grass"},
    # AFC South
    "HOU": {"name": "NRG Stadium",                "lat": 29.6847,  "lon": -95.4107,  "is_dome": 1, "surface": "turf"},
    "IND": {"name": "Lucas Oil Stadium",          "lat": 39.7601,  "lon": -86.1639,  "is_dome": 1, "surface": "turf"},
    "JAX": {"name": "EverBank Stadium",           "lat": 30.3239,  "lon": -81.6373,  "is_dome": 0, "surface": "grass"},
    "TEN": {"name": "Nissan Stadium",             "lat": 36.1665,  "lon": -86.7713,  "is_dome": 0, "surface": "turf"},
    # AFC West
    "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439,  "lon": -105.0201, "is_dome": 0, "surface": "grass"},
    "KC":  {"name": "Arrowhead Stadium",          "lat": 39.0489,  "lon": -94.4839,  "is_dome": 0, "surface": "grass"},
    "LV":  {"name": "Allegiant Stadium",          "lat": 36.0909,  "lon": -115.1833, "is_dome": 1, "surface": "turf"},
    "LAC": {"name": "SoFi Stadium",               "lat": 33.9535,  "lon": -118.3392, "is_dome": 0, "surface": "turf"},  # retractable
    # NFC East
    "DAL": {"name": "AT&T Stadium",               "lat": 32.7480,  "lon": -97.0929,  "is_dome": 1, "surface": "turf"},  # retractable roof
    "NYG": {"name": "MetLife Stadium",            "lat": 40.8135,  "lon": -74.0744,  "is_dome": 0, "surface": "turf"},
    "PHI": {"name": "Lincoln Financial Field",    "lat": 39.9007,  "lon": -75.1675,  "is_dome": 0, "surface": "grass"},
    "WAS": {"name": "Northwest Stadium",          "lat": 38.9077,  "lon": -76.8644,  "is_dome": 0, "surface": "turf"},
    # NFC North
    "CHI": {"name": "Soldier Field",              "lat": 41.8623,  "lon": -87.6167,  "is_dome": 0, "surface": "grass"},
    "DET": {"name": "Ford Field",                 "lat": 42.3400,  "lon": -83.0456,  "is_dome": 1, "surface": "turf"},
    "GB":  {"name": "Lambeau Field",              "lat": 44.5013,  "lon": -88.0622,  "is_dome": 0, "surface": "grass"},
    "MIN": {"name": "U.S. Bank Stadium",          "lat": 44.9737,  "lon": -93.2574,  "is_dome": 1, "surface": "turf"},
    # NFC South
    "ATL": {"name": "Mercedes-Benz Stadium",      "lat": 33.7554,  "lon": -84.4008,  "is_dome": 1, "surface": "turf"},
    "CAR": {"name": "Bank of America Stadium",    "lat": 35.2258,  "lon": -80.8531,  "is_dome": 0, "surface": "grass"},
    "NO":  {"name": "Caesars Superdome",          "lat": 29.9511,  "lon": -90.0812,  "is_dome": 1, "surface": "turf"},
    "TB":  {"name": "Raymond James Stadium",      "lat": 27.9759,  "lon": -82.5033,  "is_dome": 0, "surface": "grass"},
    # NFC West
    "ARI": {"name": "State Farm Stadium",         "lat": 33.5276,  "lon": -112.2626, "is_dome": 1, "surface": "grass"},  # retractable
    "LA":  {"name": "SoFi Stadium",               "lat": 33.9535,  "lon": -118.3392, "is_dome": 0, "surface": "turf"},   # retractable
    "SF":  {"name": "Levi's Stadium",             "lat": 37.4033,  "lon": -121.9694, "is_dome": 0, "surface": "grass"},
    "SEA": {"name": "Lumen Field",                "lat": 47.5952,  "lon": -122.3316, "is_dome": 0, "surface": "turf"},
}

# Indoor defaults (dome games always return these)
INDOOR_DEFAULTS = {"temp": 72, "wind": 0, "precip_prob": 0, "precip_mm": 0.0, "is_dome": 1}


class WeatherClient:
    """
    Fetches game-day weather from Open-Meteo for outdoor NFL stadiums.

    For dome / fully enclosed stadiums, always returns indoor defaults
    without making an API call.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "NFL-Predictor/1.0 (open-source project)"

    # ── Internal ─────────────────────────────────────────────────────────
    def _fetch_hourly(
        self,
        lat: float,
        lon: float,
        date_str: str,
    ) -> Optional[dict]:
        """
        Fetch hourly weather from Open-Meteo for a location on a specific date.

        Args:
            lat, lon:  Stadium coordinates
            date_str:  "YYYY-MM-DD" or full ISO datetime string

        Returns raw Open-Meteo response dict or None.
        """
        cache_key = f"weather:{lat:.4f}:{lon:.4f}:{date_str[:10]}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Determine if date is in the past (historical) or future (forecast)
        try:
            game_date = datetime.fromisoformat(date_str[:10])
            today = datetime.now(timezone.utc).replace(tzinfo=None)
            is_past = (today - game_date).days > 7
        except ValueError:
            is_past = False

        params = {
            "latitude":          lat,
            "longitude":         lon,
            "hourly":            "temperature_2m,windspeed_10m,precipitation_probability,precipitation",
            "wind_speed_unit":   "mph",
            "temperature_unit":  "fahrenheit",
            "timezone":          "auto",
            "forecast_days":     16,
        }

        if is_past:
            # Historical endpoint
            date_fmt = date_str[:10]
            params["start_date"] = date_fmt
            params["end_date"]   = date_fmt
            params.pop("forecast_days", None)
            url = "https://archive-api.open-meteo.com/v1/archive"
            ttl = TTL_WEATHER_PAST
        else:
            url = BASE_URL
            ttl = TTL_WEATHER_FORECAST

        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cache.set(cache_key, data, ttl=ttl)
            return data
        except Exception as e:
            logger.error("Open-Meteo request failed: %s", e)
            return None

    @staticmethod
    def _extract_hour(hourly: dict, game_datetime: str) -> dict:
        """
        Given hourly arrays, find the closest hour to the kickoff time
        and return temperature, wind speed and precipitation.
        """
        try:
            times = hourly.get("time", [])
            target = game_datetime[:16]  # "YYYY-MM-DDTHH:MM"

            # Find closest hour
            idx = 0
            best_diff = float("inf")
            for i, t in enumerate(times):
                diff = abs((datetime.fromisoformat(t) -
                            datetime.fromisoformat(target)).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    idx = i

            return {
                "temp":        round(hourly["temperature_2m"][idx]),
                "wind":        round(hourly["windspeed_10m"][idx]),
                "precip_prob": hourly.get("precipitation_probability", [0] * (idx + 1))[idx] or 0,
                "precip_mm":   hourly.get("precipitation", [0.0] * (idx + 1))[idx] or 0.0,
            }
        except Exception:
            return {"temp": 55, "wind": 5, "precip_prob": 0, "precip_mm": 0.0}

    # ── Public methods ────────────────────────────────────────────────────
    def get_game_weather(
        self,
        home_team: str,
        game_datetime: str = "2024-01-01T13:00",
    ) -> dict:
        """
        Get weather conditions for an NFL game based on the home team's stadium.

        Args:
            home_team:     3-letter abbreviation of the home team (e.g. "GB")
            game_datetime: ISO format string — "YYYY-MM-DDTHH:MM"
                           Defaults to a placeholder if not provided.

        Returns a dict safe to feed directly into the prediction model:
            {
                "temp":        int (°F),
                "wind":        int (mph),
                "precip_prob": int (%),
                "precip_mm":   float,
                "is_dome":     int (0 or 1),
                "stadium":     str,
            }
        """
        team = home_team.upper()
        stadium = STADIUMS.get(team)
        if not stadium:
            logger.warning("No stadium data for team: %s. Using defaults.", team)
            return {**INDOOR_DEFAULTS, "stadium": "Unknown"}

        if stadium["is_dome"]:
            return {**INDOOR_DEFAULTS, "stadium": stadium["name"]}

        raw = self._fetch_hourly(stadium["lat"], stadium["lon"], game_datetime)
        if raw is None:
            # Fallback: return plausible outdoor defaults
            return {
                "temp": 55, "wind": 7, "precip_prob": 0,
                "precip_mm": 0.0, "is_dome": 0,
                "stadium": stadium["name"],
            }

        conditions = self._extract_hour(raw["hourly"], game_datetime)
        return {
            **conditions,
            "is_dome":  0,
            "stadium":  stadium["name"],
            "surface":  stadium["surface"],
            "lat":      stadium["lat"],
            "lon":      stadium["lon"],
        }

    def get_weather_for_matchup(
        self,
        home_team: str,
        game_datetime: str,
    ) -> dict:
        """
        Alias for get_game_weather() — returns the same result.
        Provided for convenience when calling from the game predictor UI.
        """
        return self.get_game_weather(home_team, game_datetime)

    def is_bad_weather_game(self, home_team: str, game_datetime: str) -> dict[str, bool]:
        """
        Return human-readable weather risk flags for the game predictor UI.

        Returns:
            {
                "high_wind":   bool  (wind >= 20 mph),
                "freezing":    bool  (temp <= 32 °F),
                "rain":        bool  (precip_prob >= 50 %),
                "heavy_snow":  bool  (temp < 32 and precip_prob >= 40 %),
            }
        """
        w = self.get_game_weather(home_team, game_datetime)
        return {
            "high_wind":  w["wind"] >= 20,
            "freezing":   w["temp"] <= 32,
            "rain":       not w["is_dome"] and w["precip_prob"] >= 50,
            "heavy_snow": not w["is_dome"] and w["temp"] < 32 and w["precip_prob"] >= 40,
        }

    @staticmethod
    def get_stadium_info(team: str) -> Optional[dict]:
        """Return static stadium info for a team abbreviation."""
        return STADIUMS.get(team.upper())

    @staticmethod
    def outdoor_teams() -> list[str]:
        """Return list of teams with outdoor stadiums."""
        return [t for t, s in STADIUMS.items() if not s["is_dome"]]

    @staticmethod
    def dome_teams() -> list[str]:
        """Return list of teams with dome/fully indoor stadiums."""
        return [t for t, s in STADIUMS.items() if s["is_dome"]]
