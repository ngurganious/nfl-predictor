"""
prediction_history.py
=====================
Shared persistence layer for EdgeIQ prediction logging and bet tracking.

Prediction History:  auto-logged at render time → prediction_history.json
Bet Tracker:         user-logged bets → user_bets.json
"""

import json
import os
from datetime import date, datetime

HISTORY_FILE = "prediction_history.json"
BETS_FILE = "user_bets.json"


def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


# ── Prediction History ────────────────────────────────────────────────────────

def log_prediction(record: dict):
    """Upsert a prediction record by id. Preserves actual_* fields if already filled."""
    records = _load_json(HISTORY_FILE)
    gid = record.get("id")
    for i, r in enumerate(records):
        if r.get("id") == gid:
            merged = {**r, **record}
            for k in ("actual_winner", "actual_score_home", "actual_score_away",
                      "actual_total", "prediction_correct", "ou_correct"):
                if r.get(k) is not None:
                    merged[k] = r[k]
            records[i] = merged
            _save_json(HISTORY_FILE, records)
            return
    records.append(record)
    _save_json(HISTORY_FILE, records)


def load_predictions():
    return _load_json(HISTORY_FILE)


def get_daily_recs(sport: str, today_str: str = None):
    """Return today's prediction records that have a non-PASS signal."""
    if today_str is None:
        today_str = date.today().isoformat()
    return [r for r in _load_json(HISTORY_FILE)
            if r.get("sport") == sport
            and r.get("game_date") == today_str
            and r.get("kelly_signal") not in ("PASS", None)]


# ── Bet Tracker ───────────────────────────────────────────────────────────────

def log_bet(bet: dict):
    """Append or update a bet in user_bets.json, keyed by id."""
    bets = _load_json(BETS_FILE)
    bid = bet.get("id")
    for i, b in enumerate(bets):
        if b.get("id") == bid:
            bets[i] = {**b, **bet}
            _save_json(BETS_FILE, bets)
            return
    bets.append(bet)
    _save_json(BETS_FILE, bets)


def delete_bet(bet_id: str):
    bets = _load_json(BETS_FILE)
    bets = [b for b in bets if b.get("id") != bet_id]
    _save_json(BETS_FILE, bets)


def load_bets():
    return _load_json(BETS_FILE)


# ── Results backfill ──────────────────────────────────────────────────────────

def _fill_result(record, home_score, away_score):
    """Fill actual_* fields into an in-place record dict."""
    hs = int(home_score) if home_score is not None else 0
    asc = int(away_score) if away_score is not None else 0
    record["actual_score_home"] = hs
    record["actual_score_away"] = asc
    record["actual_total"] = hs + asc
    record["actual_winner"] = record["home_team"] if hs > asc else record["away_team"]
    predicted_winner = (record["home_team"] if record.get("model_home_prob", 0.5) >= 0.5
                        else record["away_team"])
    record["prediction_correct"] = (record["actual_winner"] == predicted_winner)
    if record.get("ou_lean") and record.get("ou_line") is not None:
        total = record["actual_total"]
        line = float(record["ou_line"])
        record["ou_correct"] = (total > line) if record["ou_lean"] == "OVER" else (total < line)


def backfill_results():
    """Try to fill in actual_* for past unsettled predictions. Returns count updated."""
    records = _load_json(HISTORY_FILE)
    if not records:
        return 0
    today = date.today().isoformat()
    pending = [r for r in records
               if r.get("actual_winner") is None
               and r.get("game_date", today) < today]
    if not pending:
        return 0

    updated = 0

    # --- NFL via ESPN scoreboard ---
    nfl_pending = [r for r in pending if r.get("sport") == "nfl"]
    if nfl_pending:
        try:
            from apis.espn import ESPNClient
            board = ESPNClient().get_scoreboard()
            for r in nfl_pending:
                for g in board:
                    if (g.get("home_team") == r["home_team"]
                            and g.get("away_team") == r["away_team"]
                            and g.get("status") == "STATUS_FINAL"
                            and g.get("home_score") is not None):
                        _fill_result(r, g["home_score"], g["away_score"])
                        updated += 1
                        break
        except Exception:
            pass

    # --- NHL via schedule endpoint (one call per unique past game date) ---
    nhl_pending = [r for r in pending if r.get("sport") == "nhl"]
    if nhl_pending:
        try:
            from apis.nhl import NHLClient
            nhl = NHLClient()
            unique_dates = {r["game_date"] for r in nhl_pending if r.get("game_date")}
            games_by_date = {}
            for d in unique_dates:
                data = nhl._get(f"/schedule/{d}", ttl=3600)
                if data:
                    gs = []
                    for week_block in data.get("gameWeek", []):
                        gs.extend(week_block.get("games", []))
                    games_by_date[d] = gs
            for r in nhl_pending:
                for g in games_by_date.get(r.get("game_date", ""), []):
                    home = g.get("homeTeam", {})
                    away = g.get("awayTeam", {})
                    if (home.get("abbrev") == r["home_team"]
                            and away.get("abbrev") == r["away_team"]
                            and g.get("gameState") in ("OFF", "FINAL")
                            and home.get("score") is not None):
                        _fill_result(r, home.get("score", 0), away.get("score", 0))
                        updated += 1
                        break
        except Exception:
            pass

    if updated:
        _save_json(HISTORY_FILE, records)
    return updated
