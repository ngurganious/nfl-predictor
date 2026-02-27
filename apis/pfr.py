"""
Pro Football Reference Scraper
================================
Scrapes historical game results and box scores from pro-football-reference.com.
Powers the backtesting tab and historical accuracy analysis.

Provides:
  - Full season game logs (scores, teams, stadium, conditions)
  - Individual game box scores (team and player stats)
  - 5+ years of historical data for model validation

Usage:
    from apis.pfr import PFRScraper
    scraper = PFRScraper()

    games_2023 = scraper.get_season_games(2023)
    games_5yr  = scraper.get_historical_games(years=5)

Rate limiting:  PFR blocks aggressive scrapers. This module:
  - Caches all responses for 7 days (historical data doesn't change)
  - Adds a polite 2-second delay between requests
  - Uses a realistic browser User-Agent
  - Retries once on failure with exponential backoff

Do NOT call this in a tight loop. Designed for batch data collection.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import cache

logger = logging.getLogger(__name__)

PFR_BASE    = "https://www.pro-football-reference.com"
TTL_HISTORY = 60 * 60 * 24 * 7   # 7 days — historical data is static
TTL_CURRENT = 60 * 60 * 6        # 6 hrs  — current season may update

REQUEST_DELAY = 2.5  # seconds between requests (polite scraping)

# PFR team abbreviation → our standard 3-letter abbreviation
PFR_TO_STD = {
    "GNB": "GB",  "KAN": "KC",  "NOR": "NO",  "NWE": "NE",
    "SFO": "SF",  "TAM": "TB",  "RAI": "LV",  "OAK": "LV",
    "RAM": "LA",  "SDG": "LAC", "HTX": "HOU", "CLT": "IND",
    "JAC": "JAX", "RAV": "BAL", "CRD": "ARI", "LAR": "LA",
    "LVR": "LV",  "WAS": "WAS", "PHO": "ARI",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer":         "https://www.pro-football-reference.com/",
}


def _norm(abbr: str) -> str:
    """Normalize a PFR abbreviation to our 3-letter standard."""
    if not abbr:
        return ""
    abbr = abbr.strip().upper()
    return PFR_TO_STD.get(abbr, abbr)


class PFRScraper:
    """
    Scrapes Pro Football Reference for historical NFL game data.

    Key methods:
        get_season_games(year)      → list[dict]   one row per game
        get_historical_games(years) → pd.DataFrame multi-year dataset
        get_box_score(pfr_game_id)  → dict         single game details
    """

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay   = delay
        self._last   = 0.0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Internal ─────────────────────────────────────────────────────────
    def _throttle(self):
        """Enforce polite delay between requests."""
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.time()

    def _fetch_html(self, url: str, ttl: int) -> Optional[str]:
        """Fetch page HTML with caching. Returns HTML string or None."""
        cached = cache.get(f"pfr_html:{url}")
        if cached is not None:
            return cached

        self._throttle()
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 429:
                logger.warning("PFR rate limit (429). Waiting 30s...")
                time.sleep(30)
                resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text
            cache.set(f"pfr_html:{url}", html, ttl=ttl)
            return html
        except Exception as e:
            logger.error("PFR fetch failed for %s: %s", url, e)
            return None

    # ── Season game logs ─────────────────────────────────────────────────
    def get_season_games(self, year: int) -> list[dict]:
        """
        Scrape the complete game log for one NFL season from PFR.

        Args:
            year: NFL season year (e.g. 2023 for the 2023 season)

        Returns a list of game dicts:
            season, week, game_type, gameday, home_team, away_team,
            home_score, away_score, home_win, overtime,
            winner, loser, pts_winner, pts_loser,
            yds_winner, yds_loser, to_winner, to_loser
        """
        url = f"{PFR_BASE}/years/{year}/games.htm"
        current_year = datetime.now().year
        ttl = TTL_CURRENT if year >= current_year - 1 else TTL_HISTORY

        html = self._fetch_html(url, ttl=ttl)
        if html is None:
            return []

        try:
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", id="games")
            if table is None:
                logger.warning("PFR: games table not found for year %d", year)
                return []

            rows   = table.find("tbody").find_all("tr")
            results = []

            for row in rows:
                # Skip header rows embedded in the table
                if row.get("class") and "thead" in row.get("class", []):
                    continue

                cells = row.find_all(["td", "th"])
                if not cells:
                    continue

                try:
                    data = {c.get("data-stat", ""): c for c in cells}

                    week_cell = data.get("week_num")
                    week_text = week_cell.get_text(strip=True) if week_cell else ""
                    if not week_text or week_text == "Week":
                        continue

                    # Determine game type
                    game_type = "REG"
                    if week_text in ("WildCard", "Division", "ConfChamp", "SuperBowl"):
                        game_type = "POST"

                    winner_cell = data.get("winner")
                    loser_cell  = data.get("loser")
                    if not winner_cell or not loser_cell:
                        continue

                    winner = _norm(_extract_team(winner_cell))
                    loser  = _norm(_extract_team(loser_cell))

                    # The "@" column tells us if winner was away
                    at_cell = data.get("game_location")
                    winner_was_away = (
                        at_cell is not None
                        and "@" in at_cell.get_text()
                    )
                    home_team  = loser  if winner_was_away else winner
                    away_team  = winner if winner_was_away else loser
                    home_score = _safe_int(data.get("pts_lose", data.get("pts_loser")))
                    away_score = _safe_int(data.get("pts_win",  data.get("pts_winner")))

                    if winner_was_away:
                        home_score, away_score = away_score, home_score

                    # Scores
                    pts_w = _safe_int(data.get("pts_win",  data.get("pts_winner")))
                    pts_l = _safe_int(data.get("pts_lose", data.get("pts_loser")))

                    # Yardage / turnovers
                    yds_w = _safe_int(data.get("yards_win",  data.get("yards_winner")))
                    yds_l = _safe_int(data.get("yards_lose", data.get("yards_loser")))
                    to_w  = _safe_int(data.get("to_win",     data.get("to_winner")))
                    to_l  = _safe_int(data.get("to_lose",    data.get("to_loser")))

                    gameday_cell = data.get("game_date")
                    gameday = gameday_cell.get_text(strip=True) if gameday_cell else ""

                    results.append({
                        "season":     year,
                        "week":       week_text,
                        "game_type":  game_type,
                        "gameday":    gameday,
                        "home_team":  home_team,
                        "away_team":  away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_win":   int((home_score or 0) > (away_score or 0)),
                        "overtime":   "OT" in (data.get("overtime", None) or
                                               row.get_text()),
                        "winner":     winner,
                        "loser":      loser,
                        "pts_winner": pts_w,
                        "pts_loser":  pts_l,
                        "yds_winner": yds_w,
                        "yds_loser":  yds_l,
                        "to_winner":  to_w,
                        "to_loser":   to_l,
                        # PFR game link for box score drilling
                        "pfr_link":   _extract_link(data.get("gametime") or
                                                    data.get("boxscore_word")),
                    })

                except Exception as e:
                    logger.debug("PFR: skipping row (%s)", e)
                    continue

            logger.info("PFR: scraped %d games for %d", len(results), year)
            return results

        except Exception as e:
            logger.error("PFR: failed to parse %d season: %s", year, e)
            return []

    def get_historical_games(
        self,
        years: int = 5,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Scrape multiple seasons and return a combined DataFrame.

        Args:
            years:    Number of seasons to go back (default 5)
            end_year: Last season to include (default: current year - 1)

        Returns a DataFrame with one row per game, sorted by season + week.
        Suitable for feeding directly into the backtesting tab.
        """
        if end_year is None:
            end_year = datetime.now().year - 1

        all_games = []
        for year in range(end_year - years + 1, end_year + 1):
            logger.info("PFR: fetching %d season...", year)
            games = self.get_season_games(year)
            all_games.extend(games)

        if not all_games:
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        return df.sort_values(["season", "week"]).reset_index(drop=True)

    # ── Box scores ────────────────────────────────────────────────────────
    def get_box_score(self, pfr_link: str) -> dict:
        """
        Scrape a single game box score from PFR.

        Args:
            pfr_link: Relative path from game log, e.g. "/boxscores/202401141pit.htm"

        Returns a dict with:
            game_info, home_stats, away_stats, (home_players, away_players)
        """
        if not pfr_link:
            return {}

        url = pfr_link if pfr_link.startswith("http") else f"{PFR_BASE}{pfr_link}"
        html = self._fetch_html(url, ttl=TTL_HISTORY)
        if html is None:
            return {}

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Game summary scores
            scorebox = soup.find("div", class_="scorebox")
            teams    = []
            scores   = []
            if scorebox:
                for a in scorebox.find_all("a", href=re.compile(r"/teams/")):
                    teams.append(_norm(a.text.strip()))
                for div in scorebox.find_all("div", class_="score"):
                    scores.append(_safe_int(div.text.strip()))

            # Team stats table
            team_stats = {}
            stats_table = soup.find("table", id="team_stats")
            if stats_table:
                rows = stats_table.find_all("tr")
                if len(rows) >= 3:
                    headers  = [td.get_text(strip=True) for td in rows[0].find_all("th")]
                    visitor_vals = [td.get_text(strip=True) for td in rows[1].find_all("td")]
                    home_vals    = [td.get_text(strip=True) for td in rows[2].find_all("td")]
                    if teams:
                        team_stats[teams[0]] = dict(zip(headers[1:], visitor_vals))
                        team_stats[teams[1] if len(teams) > 1 else "home"] = \
                            dict(zip(headers[1:], home_vals))

            result = {
                "url":        url,
                "teams":      teams,
                "scores":     scores,
                "team_stats": team_stats,
            }
            return result

        except Exception as e:
            logger.error("PFR: box score parse failed for %s: %s", url, e)
            return {}

    # ── Convenience ──────────────────────────────────────────────────────
    def build_backtest_dataset(self, years: int = 5) -> pd.DataFrame:
        """
        Build the historical dataset used in the backtesting tab.

        Returns a DataFrame matching the structure of games_processed.csv
        so it can be merged with our existing model pipeline:
            season, week, game_type, gameday, home_team, away_team,
            home_score, away_score, home_win
        """
        df = self.get_historical_games(years=years)
        if df.empty:
            return df

        # Rename to match games_processed column names
        df = df.rename(columns={
            "gameday": "gameday",
        })

        # Numeric week for sorting (playoffs come after week 18)
        playoff_order = {
            "WildCard": 19, "Division": 20, "ConfChamp": 21, "SuperBowl": 22
        }
        df["week_num"] = df["week"].apply(
            lambda w: playoff_order.get(w, _safe_int(w) or 0)
        )

        return df.sort_values(["season", "week_num"]).reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _extract_team(cell) -> str:
    """Pull team abbreviation from a PFR table cell (may be inside <a> tag)."""
    if cell is None:
        return ""
    a = cell.find("a")
    if a:
        href = a.get("href", "")
        m = re.search(r"/teams/([a-z]{2,3})/", href)
        if m:
            return m.group(1)
        return a.get_text(strip=True)
    return cell.get_text(strip=True)


def _extract_link(cell) -> str:
    """Extract href from a table cell."""
    if cell is None:
        return ""
    a = cell.find("a")
    return a.get("href", "") if a else ""


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    text = val.get_text(strip=True) if hasattr(val, "get_text") else str(val)
    text = re.sub(r"[^\d\-]", "", text)
    try:
        return int(text)
    except (ValueError, TypeError):
        return None
