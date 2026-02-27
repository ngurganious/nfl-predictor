"""
NFL Predictor API Integration Layer
====================================
Modules:
    tank01    - Tank01 NFL API (RapidAPI) — primary live data feed
    espn      - ESPN unofficial API       — injuries & rosters
    weather   - Open-Meteo API            — stadium weather
    pfr       - Pro Football Reference    — historical box scores
    odds      - The Odds API              — Vegas lines & spreads
"""

from .tank01 import Tank01Client
from .espn import ESPNClient
from .weather import WeatherClient
from .pfr import PFRScraper
from .odds import OddsClient

__all__ = [
    "Tank01Client",
    "ESPNClient",
    "WeatherClient",
    "PFRScraper",
    "OddsClient",
]
