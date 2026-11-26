from datetime import date, datetime, timezone

import pytest

from src.data_pipeline import kalshi, storage
from src.portfolio_analysis import odds
from src.portfolio_analysis.odds import OddsEvent, Quote

CLOSE = datetime(2026, 12, 9, 19, tzinfo=timezone.utc)


def _fed(hike=0.92, hold=0.05):
    return OddsEvent("KXFEDDECISION", "KXFEDDECISION-26DEC", "Fed decision in Dec 2026?", CLOSE,
                     "choice", [Quote("Cut 25bps", 0.01), Quote("Fed maintains rate", hold),
                                Quote("Hike 25bps", hike)])


def _cpi(probs=(0.99, 0.9, 0.8, 0.4, 0.1, 0.02)):
    strikes = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
    return OddsEvent("KXCPIYOY", "KXCPIYOY-26NOV", "CPI YoY in November", CLOSE, "ladder",
                     [Quote(f"Above {k}%", p, k) for k, p in zip(strikes, probs)])


def test_quote_prob_uses_the_mid_unless_the_spread_is_wide():
    assert odds.quote_prob("0.92", "0.93", "0.90") == pytest.approx(0.925)
    assert odds.quote_prob("0.10", "0.60", "0.40") == 0.40
    assert odds.quote_prob("0", "0", "0") is None
    assert odds.quote_prob(None, "", None) is None


def test_ladder_median_and_range():
    e = _cpi()
    assert odds.median(e) == pytest.approx(3.275)         # 0.8 at 3.2, 0.4 at 3.3
    assert odds.summary(e) == "median 3.3%, 80% range 3.1% to 3.4%"
    assert odds.crossing(_cpi((0.99, 0.98, 0.97, 0.96, 0.95, 0.9)).quotes, 0.5) == (3.5, "above")
    assert odds.crossing(_cpi((0.4, 0.3, 0.2, 0.1, 0.05, 0.01)).quotes, 0.5) == (3.0, "below")
    assert "over 3.5%" in odds.summary(_cpi((0.99, 0.98, 0.97, 0.96, 0.95, 0.9)))


def test_choice_summary_hides_long_shots():
    assert odds.summary(_fed()) == "Hike 25bps 92%, Fed maintains rate 5%"
    single = OddsEvent("KXRECSSNBER", "KXRECSSNBER-27", "Recession in 2027?", None, "choice",
                       [Quote("Yes", 0.22)])
    assert odds.summary(single) == "22% yes"


def test_change_reports_big_moves_only():
    before = _fed(hike=0.70, hold=0.27).quotes
    assert odds.change(_fed(), before) == "Hike 25bps up from 70%"
    assert odds.change(_fed(), _fed(hike=0.90, hold=0.07).quotes) == ""
    assert odds.change(_cpi(), _cpi((0.99, 0.95, 0.9, 0.8, 0.4, 0.1)).quotes) == "median was 3.4%"
    assert odds.change(_fed(), None) == ""


def test_render():
    before = {"KXFEDDECISION-26DEC": (date(2026, 11, 19), _fed(hike=0.70, hold=0.27).quotes)}
    md = odds.render([_fed(), _cpi()], before, date(2026, 11, 26))
    assert md.startswith("## Market odds")
    assert ("- **Fed decision in Dec 2026?** (Dec 09): Hike 25bps 92%, Fed maintains rate 5%; "
            "Hike 25bps up from 70% 7 days ago") in md
    assert "**CPI YoY in November**" in md
    assert "No open markets" in odds.render([])


def test_parse_kalshi_events():
    ladder = {"event_ticker": "KXU3-26NOV", "title": "Unemployment in November", "markets": [
        {"status": "active", "strike_type": "greater", "floor_strike": 4.0,
         "yes_sub_title": "Above 4.0%", "yes_bid_dollars": "0.85", "yes_ask_dollars": "0.86",
         "close_time": "2026-12-04T13:29:00Z"},
        {"status": "active", "strike_type": "greater", "floor_strike": 4.1,
         "yes_sub_title": "Above 4.1%", "yes_bid_dollars": "0.45", "yes_ask_dollars": "0.49",
         "close_time": "2026-12-04T13:29:00Z"},
        {"status": "finalized", "strike_type": "greater", "floor_strike": 4.2,
         "yes_sub_title": "Above 4.2%", "yes_bid_dollars": "0.1", "yes_ask_dollars": "0.2"}]}
    e = kalshi.parse_event(ladder, "KXU3")
    assert e.kind == "ladder" and [q.strike for q in e.quotes] == [4.0, 4.1]
    assert e.closes == datetime(2026, 12, 4, 13, 29, tzinfo=timezone.utc)
    assert e.unit == "%"

    choice = {"event_ticker": "KXFEDDECISION-26DEC", "title": "Fed decision?", "markets": [
        {"status": "active", "strike_type": "custom", "yes_sub_title": "Hike 25bps",
         "yes_bid_dollars": "0.92", "yes_ask_dollars": "0.93"}]}
    assert kalshi.parse_event(choice, "KXFEDDECISION").kind == "choice"
    assert kalshi.parse_event({"event_ticker": "X", "markets": []}, "X") is None


def test_odds_are_stored_by_day(tmp_path):
    db = tmp_path / "x.db"
    storage.save_odds([_fed(hike=0.70)], date(2026, 11, 19), db)
    storage.save_odds([_fed(hike=0.80)], date(2026, 11, 20), db)
    storage.save_odds([_fed(hike=0.81)], date(2026, 11, 20), db)   # same day replaces
    day, quotes = storage.load_odds_before("KXFEDDECISION-26DEC", date(2026, 11, 20), db)
    assert day == date(2026, 11, 20)
    assert {q.outcome: q.prob for q in quotes}["Hike 25bps"] == 0.81
    assert storage.load_odds_before("KXFEDDECISION-26DEC", date(2026, 11, 18), db) is None
