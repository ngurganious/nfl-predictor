"""
The Odds API Client
=====================
Free tier: 500 requests / month (no auto-refresh needed).
https://the-odds-api.com/liveapi/guides/v4/

Provides:
  - Vegas moneylines (h2h)
  - Spreads
  - Over/Unders (totals)

Usage:
    from apis.odds import OddsClient
    client = OddsClient(api_key="YOUR_ODDS_API_KEY")

    all_odds  = client.get_nfl_odds()
    game_odds = client.get_game_odds("KC", "BUF")

Set ODDS_API_KEY in your .env file or pass it to the constructor.

Free tier notes:
  - 500 requests/month (roughly one call per 1.5 hours)
  - This client caches for 4 hours to stay comfortably within limits
  - Remaining quota is logged after each successful call
  - Stale data (>4h) falls back to the cached value rather than erroring
"""

import logging
import os
from typing import Optional

import requests

from . import cache

logger = logging.getLogger(__name__)

BASE_URL  = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

TTL_ODDS = 60 * 60 * 4  # 4 hrs — balances freshness vs. 500 req/month limit

# Bookmaker preference order (most accurate lines first)
PREFERRED_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars",
    "pointsbetus", "williamhill_us", "bovada",
]

# Team name → our 3-letter abbreviation
# The Odds API returns full city/team names; we need to map them back.
NAME_TO_ABV = {
    "Arizona Cardinals":       "ARI",
    "Atlanta Falcons":         "ATL",
    "Baltimore Ravens":        "BAL",
    "Buffalo Bills":           "BUF",
    "Carolina Panthers":       "CAR",
    "Chicago Bears":           "CHI",
    "Cincinnati Bengals":      "CIN",
    "Cleveland Browns":        "CLE",
    "Dallas Cowboys":          "DAL",
    "Denver Broncos":          "DEN",
    "Detroit Lions":           "DET",
    "Green Bay Packers":       "GB",
    "Houston Texans":          "HOU",
    "Indianapolis Colts":      "IND",
    "Jacksonville Jaguars":    "JAX",
    "Kansas City Chiefs":      "KC",
    "Las Vegas Raiders":       "LV",
    "Los Angeles Chargers":    "LAC",
    "Los Angeles Rams":        "LA",
    "Miami Dolphins":          "MIA",
    "Minnesota Vikings":       "MIN",
    "New England Patriots":    "NE",
    "New Orleans Saints":      "NO",
    "New York Giants":         "NYG",
    "New York Jets":           "NYJ",
    "Philadelphia Eagles":     "PHI",
    "Pittsburgh Steelers":     "PIT",
    "San Francisco 49ers":     "SF",
    "Seattle Seahawks":        "SEA",
    "Tampa Bay Buccaneers":    "TB",
    "Tennessee Titans":        "TEN",
    "Washington Commanders":   "WAS",
}

ABV_TO_NAME = {v: k for k, v in NAME_TO_ABV.items()}

# ── NHL team name → abbreviation mapping ─────────────────────────────────────
NHL_NAME_TO_ABV = {
    "Anaheim Ducks":         "ANA",
    "Boston Bruins":         "BOS",
    "Buffalo Sabres":        "BUF",
    "Calgary Flames":        "CGY",
    "Carolina Hurricanes":   "CAR",
    "Chicago Blackhawks":    "CHI",
    "Colorado Avalanche":    "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars":          "DAL",
    "Detroit Red Wings":     "DET",
    "Edmonton Oilers":       "EDM",
    "Florida Panthers":      "FLA",
    "Los Angeles Kings":     "LAK",
    "Minnesota Wild":        "MIN",
    "Montreal Canadiens":    "MTL",
    "Montréal Canadiens":    "MTL",
    "Nashville Predators":   "NSH",
    "New Jersey Devils":     "NJD",
    "New York Islanders":    "NYI",
    "New York Rangers":      "NYR",
    "Ottawa Senators":       "OTT",
    "Philadelphia Flyers":   "PHI",
    "Pittsburgh Penguins":   "PIT",
    "San Jose Sharks":       "SJS",
    "Seattle Kraken":        "SEA",
    "St. Louis Blues":       "STL",
    "Tampa Bay Lightning":   "TBL",
    "Toronto Maple Leafs":   "TOR",
    "Utah Hockey Club":      "UTA",
    "Utah Mammoth":          "UTA",
    "Vancouver Canucks":     "VAN",
    "Vegas Golden Knights":  "VGK",
    "Washington Capitals":   "WSH",
    "Winnipeg Jets":         "WPG",
}

NHL_ABV_TO_NAME = {v: k for k, v in NHL_NAME_TO_ABV.items()}

NHL_SPORT_KEY = "icehockey_nhl"


class OddsClient:
    """
    Wrapper around The Odds API v4 for NFL Vegas lines.

    All methods return plain Python structures (lists / dicts).
    Quota usage is printed to log after every live API call.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "ODDS_API_KEY not set. Odds calls will fail. "
                "Add it to your .env file."
            )
        self.session = requests.Session()
        self._last_used = None
        self._last_remaining = None

    # ── Internal ─────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict, ttl: int) -> Optional[dict]:
        if not self.api_key:
            logger.error("Cannot call Odds API — ODDS_API_KEY missing.")
            return None

        cache_key = f"odds:{endpoint}:{sorted(params.items())}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        params["apiKey"] = self.api_key
        url = f"{BASE_URL}/{endpoint}"

        try:
            resp = self.session.get(url, params=params, timeout=10)

            # Log and store remaining quota from response headers
            remaining = resp.headers.get("x-requests-remaining", "?")
            used      = resp.headers.get("x-requests-used", "?")
            logger.info("Odds API quota — used: %s, remaining: %s", used, remaining)
            try:
                self._last_used = int(used)
                self._last_remaining = int(remaining)
            except (TypeError, ValueError):
                pass

            if resp.status_code == 429:
                logger.warning("Odds API rate limit hit. Returning cached or None.")
                return None

            resp.raise_for_status()
            data = resp.json()
            # Don't cache the API key in the payload
            params.pop("apiKey", None)
            cache.set(cache_key, data, ttl=ttl)
            return data

        except requests.HTTPError as e:
            logger.error("Odds API HTTP error on %s: %s", endpoint, e)
        except Exception as e:
            logger.error("Odds API request failed on %s: %s", endpoint, e)
        return None

    # ── Public methods ────────────────────────────────────────────────────
    def get_nfl_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> list[dict]:
        """
        Fetch all upcoming NFL game odds.

        Args:
            markets:     Comma-separated market types:
                           h2h (moneyline), spreads, totals (over/under)
            regions:     "us" for US bookmakers
            odds_format: "american" (+/-) or "decimal"

        Returns a list of game dicts, each normalised to:
            {
                game_id, home_team, away_team, commence_time,
                spread:      {"home": float, "away": float, "book": str},
                moneyline:   {"home": int,   "away": int,   "book": str},
                total:       {"line": float, "book": str},
            }
        """
        raw = self._get(
            f"sports/{SPORT_KEY}/odds",
            {"regions": regions, "markets": markets, "oddsFormat": odds_format},
            ttl=TTL_ODDS,
        )
        if raw is None:
            return []

        results = []
        for event in raw:
            try:
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                home_abv  = NAME_TO_ABV.get(home_name, home_name)
                away_abv  = NAME_TO_ABV.get(away_name, away_name)

                parsed = {
                    "game_id":      event.get("id", ""),
                    "home_team":    home_abv,
                    "away_team":    away_abv,
                    "home_name":    home_name,
                    "away_name":    away_name,
                    "commence_time": event.get("commence_time", ""),
                    "spread":       None,
                    "moneyline":    None,
                    "total":        None,
                }

                bookmakers = event.get("bookmakers", [])
                parsed["spread"]    = _extract_spread(bookmakers, home_abv, away_abv)
                parsed["moneyline"] = _extract_moneyline(bookmakers, home_abv, away_abv)
                parsed["total"]     = _extract_total(bookmakers)

                results.append(parsed)
            except Exception as e:
                logger.debug("Odds: skipping event — %s", e)
                continue

        return results

    def get_game_odds(self, home_team: str, away_team: str) -> Optional[dict]:
        """
        Find odds for a specific matchup.

        Args:
            home_team: 3-letter abbreviation (e.g. "KC")
            away_team: 3-letter abbreviation (e.g. "BUF")

        Returns the matching game dict from get_nfl_odds(), or None if
        the game hasn't been listed yet.
        """
        all_odds = self.get_nfl_odds()
        for game in all_odds:
            if game["home_team"] == home_team and game["away_team"] == away_team:
                return game
            # Also check flipped (in case the game is listed home/away differently)
            if game["home_team"] == away_team and game["away_team"] == home_team:
                return game
        return None

    def get_spread(self, home_team: str, away_team: str) -> Optional[float]:
        """
        Return the Vegas spread for a game (from home team's perspective).
        Negative = home team favored.

        Returns None if game not found.
        """
        game = self.get_game_odds(home_team, away_team)
        if game and game["spread"]:
            return game["spread"].get("home")
        return None

    def get_total(self, home_team: str, away_team: str) -> Optional[float]:
        """Return the Vegas over/under total for a game."""
        game = self.get_game_odds(home_team, away_team)
        if game and game["total"]:
            return game["total"].get("line")
        return None

    def get_implied_probability(
        self, home_team: str, away_team: str
    ) -> Optional[dict[str, float]]:
        """
        Convert moneyline odds to implied win probabilities.

        Returns:
            {"home": 0.62, "away": 0.38} or None if unavailable.
        """
        game = self.get_game_odds(home_team, away_team)
        if not game or not game["moneyline"]:
            return None

        home_ml = game["moneyline"].get("home")
        away_ml = game["moneyline"].get("away")

        if home_ml is None or away_ml is None:
            return None

        home_prob = _american_to_prob(home_ml)
        away_prob = _american_to_prob(away_ml)

        # Remove vig (normalize to 100%)
        total = home_prob + away_prob
        if total == 0:
            return None

        return {
            "home": round(home_prob / total, 4),
            "away": round(away_prob / total, 4),
        }

    def get_formatted_lines(self, home_team: str, away_team: str) -> str:
        """
        Return a human-readable string of the Vegas lines for display in the UI.

        Example: "KC -6.5 | O/U 47.5 | ML: KC -280 / BUF +230"
        """
        game = self.get_game_odds(home_team, away_team)
        if not game:
            return "Lines not available"

        parts = []
        if game["spread"]:
            home_sp = game["spread"].get("home", 0)
            sign    = "+" if home_sp > 0 else ""
            parts.append(f"{home_team} {sign}{home_sp:.1f}")

        if game["total"]:
            parts.append(f"O/U {game['total'].get('line', 0):.1f}")

        if game["moneyline"]:
            home_ml = game["moneyline"].get("home", "")
            away_ml = game["moneyline"].get("away", "")
            h_sign  = "+" if isinstance(home_ml, (int, float)) and home_ml > 0 else ""
            a_sign  = "+" if isinstance(away_ml, (int, float)) and away_ml > 0 else ""
            parts.append(f"ML: {home_team} {h_sign}{home_ml} / {away_team} {a_sign}{away_ml}")

        return " | ".join(parts) if parts else "Lines not available"

    def get_nhl_odds(
        self,
        markets: str = "h2h,totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> list:
        """
        Fetch all upcoming NHL game odds.
        Uses 'icehockey_nhl' sport key. Same structure as get_nfl_odds().

        Returns list of game dicts with moneyline and total (puck lines not standard).
        """
        raw = self._get(
            f"sports/{NHL_SPORT_KEY}/odds",
            {"regions": regions, "markets": markets, "oddsFormat": odds_format},
            ttl=TTL_ODDS,
        )
        if raw is None:
            return []

        results = []
        for event in raw:
            try:
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                home_abv  = NHL_NAME_TO_ABV.get(home_name, home_name)
                away_abv  = NHL_NAME_TO_ABV.get(away_name, away_name)

                bookmakers = event.get("bookmakers", [])

                # Extract moneyline (NHL uses h2h, not spreads)
                ml = None
                total = None
                book = _pick_book(bookmakers)
                if book:
                    for market in book.get("markets", []):
                        if market["key"] == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                            ml = {
                                "home": outcomes.get(home_name),
                                "away": outcomes.get(away_name),
                                "book": book["key"],
                            }
                        elif market["key"] == "totals":
                            for outcome in market.get("outcomes", []):
                                if outcome["name"] == "Over":
                                    total = {"line": outcome["point"], "book": book["key"]}

                results.append({
                    "game_id":      event.get("id", ""),
                    "home_team":    home_abv,
                    "away_team":    away_abv,
                    "home_name":    home_name,
                    "away_name":    away_name,
                    "commence_time": event.get("commence_time", ""),
                    "moneyline":    ml,
                    "total":        total,
                    "spread":       None,  # NHL doesn't use point spreads
                })
            except Exception as e:
                logger.debug("NHL odds: skipping event — %s", e)
                continue

        return results

    def get_nhl_game_odds(self, home_team: str, away_team: str) -> Optional[dict]:
        """Find NHL odds for a specific matchup by team abbreviation."""
        all_odds = self.get_nhl_odds()
        for game in all_odds:
            if game["home_team"] == home_team and game["away_team"] == away_team:
                return game
            if game["home_team"] == away_team and game["away_team"] == home_team:
                return game
        return None

    def check_quota(self) -> Optional[dict]:
        """
        Make a cheap call to check remaining quota without fetching game data.
        Returns {"used": int, "remaining": int} or None.
        """
        if not self.api_key:
            return None
        try:
            url  = f"{BASE_URL}/sports"
            resp = self.session.get(
                url, params={"apiKey": self.api_key}, timeout=10
            )
            remaining = resp.headers.get("x-requests-remaining")
            used      = resp.headers.get("x-requests-used")
            return {
                "used":      int(used)      if used      else None,
                "remaining": int(remaining) if remaining else None,
            }
        except Exception as e:
            logger.error("Odds API quota check failed: %s", e)
            return None

    def get_quota_cached(self):
        if self._last_used is not None:
            return {"used": self._last_used, "remaining": self._last_remaining}
        return None

    # ── Player Prop Markets ──────────────────────────────────────────────

    PROP_MARKETS = "player_pass_yds,player_rush_yds,player_reception_yds"

    def get_nfl_events(self):
        raw = self._get(
            f"sports/{SPORT_KEY}/events",
            {"dateFormat": "iso"},
            ttl=TTL_ODDS,
        )
        if raw is None:
            return []
        results = []
        for event in raw:
            home_abv = NAME_TO_ABV.get(event.get("home_team", ""), "")
            away_abv = NAME_TO_ABV.get(event.get("away_team", ""), "")
            results.append({
                "event_id": event.get("id", ""),
                "home_team": home_abv,
                "away_team": away_abv,
                "commence_time": event.get("commence_time", ""),
            })
        return results

    def get_player_props(self, event_id, markets=None):
        if markets is None:
            markets = self.PROP_MARKETS
        raw = self._get(
            f"sports/{SPORT_KEY}/events/{event_id}/odds",
            {"regions": "us", "markets": markets, "oddsFormat": "american"},
            ttl=TTL_ODDS,
        )
        if raw is None:
            return []
        bookmakers = raw.get("bookmakers", [])
        book = _pick_book(bookmakers)
        if not book:
            return []
        props = []
        for market in book.get("markets", []):
            market_key = market["key"]
            for outcome in market.get("outcomes", []):
                props.append({
                    "player": outcome.get("description", ""),
                    "market": market_key,
                    "name": outcome.get("name", ""),
                    "line": outcome.get("point"),
                    "odds": outcome.get("price"),
                    "book": book["key"],
                })
        return props


# ── Helpers ──────────────────────────────────────────────────────────────────
def _pick_book(bookmakers: list, preferred: list = PREFERRED_BOOKS) -> Optional[dict]:
    """Return the first bookmaker matching preference order."""
    book_map = {b["key"]: b for b in bookmakers}
    for key in preferred:
        if key in book_map:
            return book_map[key]
    return bookmakers[0] if bookmakers else None


def _extract_spread(
    bookmakers: list, home_abv: str, away_abv: str
) -> Optional[dict]:
    book = _pick_book(bookmakers)
    if not book:
        return None
    for market in book.get("markets", []):
        if market["key"] == "spreads":
            outcomes = {o["name"]: o["point"] for o in market.get("outcomes", [])}
            home_name = ABV_TO_NAME.get(home_abv, home_abv)
            away_name = ABV_TO_NAME.get(away_abv, away_abv)
            return {
                "home": outcomes.get(home_name),
                "away": outcomes.get(away_name),
                "book": book["key"],
            }
    return None


def _extract_moneyline(
    bookmakers: list, home_abv: str, away_abv: str
) -> Optional[dict]:
    book = _pick_book(bookmakers)
    if not book:
        return None
    for market in book.get("markets", []):
        if market["key"] == "h2h":
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            home_name = ABV_TO_NAME.get(home_abv, home_abv)
            away_name = ABV_TO_NAME.get(away_abv, away_abv)
            return {
                "home": outcomes.get(home_name),
                "away": outcomes.get(away_name),
                "book": book["key"],
            }
    return None


def _extract_total(bookmakers: list) -> Optional[dict]:
    book = _pick_book(bookmakers)
    if not book:
        return None
    for market in book.get("markets", []):
        if market["key"] == "totals":
            for outcome in market.get("outcomes", []):
                if outcome["name"] == "Over":
                    return {"line": outcome["point"], "book": book["key"]}
    return None


def _american_to_prob(ml: float) -> float:
    """Convert American moneyline odds to implied probability."""
    if ml is None:
        return 0.5
    try:
        ml = float(ml)
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)
    except (TypeError, ValueError):
        return 0.5
