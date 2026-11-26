"""Command line entry point.

    python -m src.cli snapshot                    # pull from Robinhood
    python -m src.cli snapshot --csv export.csv   # rebuild from a Robinhood activity CSV
    python -m src.cli history                     # list saved snapshots
    python -m src.cli position NVDA               # deep dive from the latest snapshot
    python -m src.cli news                        # fetch and show news for your holdings
    python -m src.cli options                     # implied moves and IV for your holdings
    python -m src.cli catalysts                   # upcoming events for your holdings
    python -m src.cli report daily                # write today's report to reports/daily/
    python -m src.cli report weekly               # this week so far, to reports/weekly/
    python -m src.cli report alerts               # what needs attention today
    python -m src.cli insiders                    # notable insider buying and selling
    python -m src.cli macro                       # rates and inflation backdrop
    python -m src.cli earnings-call NFLX          # summary of the latest earnings call
    python -m src.cli ideas                       # add / hold / trim / sell, watchlist, cash plan
    python -m src.cli odds                        # Fed, CPI and recession odds from Kalshi
    python -m src.cli run                         # all of the above, then notify (for the scheduler)
    python -m src.cli notify --test               # check your notification channels
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.data_pipeline import storage
from src.data_pipeline.market_data import (
    apply_price_stats, benchmark_day_change, enrich_holdings, price_history,
)
from src.data_pipeline.models import Holding, PortfolioSnapshot
from src.portfolio_analysis import metrics
from src.portfolio_analysis.technicals import trend_label

console = Console()


def _pct(x: float | None) -> str:
    if x is None:
        return "-"
    color = "green" if x >= 0 else "red"
    return f"[{color}]{x:+.2%}[/{color}]"


def _money(x: float | None) -> str:
    return "-" if x is None else f"${x:,.2f}"


def _num(x: float | None, fmt: str = ".1f") -> str:
    return "-" if x is None else format(x, fmt)


def _plain_pct(x: float | None, fmt: str = ".0%") -> str:
    return "-" if x is None else format(x, fmt)


def _big(x: float | None) -> str:
    if x is None:
        return "-"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= size:
            return f"${x / size:.1f}{unit}"
    return f"${x:,.0f}"


def render(snap: PortfolioSnapshot, cfg: dict, closes: pd.DataFrame | None = None) -> None:
    w = metrics.weights(snap)
    table = Table(title=f"Holdings ({snap.source}, {snap.taken_at:%Y-%m-%d %H:%M})")
    for col in ("Ticker", "Shares", "Avg cost", "Price", "Value", "Weight",
                "Day", "Total P/L", "52w pos", "Next earnings"):
        table.add_column(col, justify="left" if col == "Ticker" else "right")
    for h in sorted(snap.holdings, key=lambda h: -h.market_value):
        table.add_row(
            h.ticker, f"{h.quantity:g}", _money(h.average_cost), _money(h.price),
            _money(h.market_value), f"{w.get(h.ticker, 0):.1%}", _pct(h.day_change_pct),
            f"{_money(h.unrealized_pl)} {_pct(h.unrealized_pl_pct)}",
            "-" if h.week52_position is None else f"{h.week52_position:.0%}",
            f"{h.next_earnings:%b %d}" if h.next_earnings else "-",
        )
    console.print(table)
    render_health(snap)

    if snap.options:
        opt = Table(title="Options")
        for col in ("Contract", "Qty", "Avg cost", "Mark", "Value", "DTE", "IV", "Delta"):
            opt.add_column(col, justify="right")
        for o in snap.options:
            opt.add_row(
                f"{o.underlying} {o.expiration:%m/%d} {o.strike:g}{o.option_type[:1].upper()}",
                f"{o.contracts:g}", _money(o.average_cost), _money(o.mark_price),
                _money(o.market_value), str(o.days_to_expiry),
                "-" if o.implied_volatility is None else f"{o.implied_volatility:.0%}",
                "-" if o.delta is None else f"{o.delta:.2f}",
            )
        console.print(opt)

    cash_note = " (estimated from CSV)" if snap.account.cash_is_estimate else ""
    console.print(f"Holdings {_money(snap.holdings_value)} + cash {_money(snap.account.cash)}"
                  f"{cash_note} = [bold]{_money(snap.total_equity)}[/bold]")

    bench = benchmark_day_change(cfg["benchmarks"])
    console.print("Today: portfolio " + _pct(metrics.portfolio_day_change(snap)) + "  " +
                  "  ".join(f"{sym} {_pct(v)}" for sym, v in bench.items()))

    sectors = metrics.sector_weights(snap)
    console.print("Sectors: " + ", ".join(f"{s} {v:.0%}" for s, v in sectors.items()))
    if closes is not None:
        render_risk_and_tax(snap, closes, cfg)

    flags = metrics.risk_flags(snap, cfg["risk"])
    if flags:
        console.print("\n[bold yellow]Flags[/bold yellow]")
        for f in flags:
            console.print(f"  - {f}")


def render_health(snap: PortfolioSnapshot) -> None:
    table = Table(title="Position health")
    for col in ("Ticker", "1W", "1M", "3M", "YTD", "1Y", "Vol", "Max DD", "RSI", "Trend",
                "Fwd P/E", "Rev gr.", "Short %", "Analyst", "Upside"):
        table.add_column(col, justify="left" if col == "Ticker" else "right")
    for h in sorted(snap.holdings, key=lambda h: -h.market_value):
        table.add_row(
            h.ticker, _pct(h.return_1w), _pct(h.return_1m), _pct(h.return_3m),
            _pct(h.return_ytd), _pct(h.return_1y), _plain_pct(h.volatility_1y),
            _plain_pct(h.max_drawdown_1y), _num(h.rsi_14, ".0f"),
            trend_label(h.price, h.sma_50, h.sma_200), _num(h.forward_pe),
            _plain_pct(h.revenue_growth), _plain_pct(h.short_pct_float, ".1%"),
            h.analyst_rating or "-", _pct(h.analyst_upside),
        )
    console.print(table)


def render_risk_and_tax(snap: PortfolioSnapshot, closes: pd.DataFrame, cfg: dict) -> None:
    risk = metrics.portfolio_risk(snap, closes, cfg["benchmarks"][0],
                                  cfg["risk"]["correlation_threshold"])
    console.print(f"Portfolio beta vs {cfg['benchmarks'][0]}: {_num(risk.beta, '.2f')}   "
                  f"annualized volatility: {_plain_pct(risk.volatility)}")
    if risk.correlated_pairs:
        console.print("Moves together (1Y daily correlation): " + ", ".join(
            f"{a}/{b} {c:.2f}" for a, b, c in risk.correlated_pairs))

    year = date.today().year
    gains = metrics.realized_gains(snap.transactions, year)
    st = sum(g.gain for g in gains if not g.long_term)
    lt = sum(g.gain for g in gains if g.long_term)
    console.print(f"{year} realized: short-term {_money(st)}, long-term {_money(lt)}   "
                  f"dividends {_money(metrics.dividend_income(snap.transactions, year))}")


def render_performance(snap: PortfolioSnapshot, cfg: dict) -> None:
    flows = metrics.cash_flows(snap.transactions)
    if not flows:
        return
    today = snap.taken_at.date()
    net_in = sum(amt for _, amt in flows)
    mwr = metrics.money_weighted_return(flows, snap.total_equity, today)
    console.print(f"Since first deposit {flows[0][0]:%Y-%m-%d}: net deposits {_money(net_in)}, "
                  f"now {_money(snap.total_equity)} ({_money(snap.total_equity - net_in)} gain), "
                  f"{_pct(mwr)}/yr money-weighted")

    closes = price_history(cfg["benchmarks"], start=flows[0][0])
    same_money = []
    for sym in cfg["benchmarks"]:
        value = metrics.benchmark_equivalent(flows, closes[sym]) if sym in closes else None
        if value is not None:
            rate = metrics.money_weighted_return(flows, value, today)
            same_money.append(f"{sym} {_money(value)} ({_pct(rate)}/yr)")
    if same_money:
        console.print("Same deposits into: " + ", ".join(same_money))


def cmd_snapshot(args, cfg: dict) -> None:
    if args.csv:
        from src.data_pipeline.sources.csv_import import CsvSource
        source = CsvSource(args.csv)
    else:
        from src.data_pipeline.sources.robinhood import RobinhoodSource
        rh = cfg.get("robinhood", {})
        source = RobinhoodSource(session_days=rh.get("session_days", 30),
                                 login_timeout_min=rh.get("login_timeout_min", 10))

    with console.status(f"Fetching portfolio from {source.name}..."):
        snap = source.fetch_snapshot()
    with console.status("Fetching market data..."):
        enrich_holdings(snap.holdings)
        closes = price_history([h.ticker for h in snap.holdings] + cfg["benchmarks"])
        apply_price_stats(snap.holdings, closes, cfg["benchmarks"][0])

    render(snap, cfg, closes)
    render_performance(snap, cfg)
    if not args.no_save:
        snapshot_id = storage.save_snapshot(snap, Path(cfg["database"]))
        console.print(f"\nSaved snapshot #{snapshot_id} to {cfg['database']}")


def cmd_history(args, cfg: dict) -> None:
    rows = storage.list_snapshots(args.limit, Path(cfg["database"]))
    table = Table(title="Snapshots")
    for col in ("#", "Taken at", "Source", "Holdings", "Cash", "Total"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(str(r["id"]), r["taken_at"], r["source"], _money(r["holdings_value"]),
                      _money(r["cash"]), _money(r["total_equity"]))
    console.print(table)


def cmd_position(args, cfg: dict) -> None:
    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return
    ticker = args.ticker.upper()
    h: Holding | None = next((h for h in snap.holdings if h.ticker == ticker), None)
    if h is None:
        console.print(f"{ticker} isn't in your latest snapshot ({snap.taken_at:%Y-%m-%d}).")
        return

    weight = metrics.weights(snap).get(ticker, 0)
    console.rule(f"{h.ticker}  {h.name or ''}  ({h.sector or '-'} / {h.industry or '-'})")
    sections = {
        "Position": [
            ("Shares", f"{h.quantity:g}"), ("Avg cost", _money(h.average_cost)),
            ("Price", _money(h.price)), ("Value", _money(h.market_value)),
            ("Weight", f"{weight:.1%}"),
            ("Unrealized", f"{_money(h.unrealized_pl)} {_pct(h.unrealized_pl_pct)}"),
        ],
        "Price behavior": [
            ("1W / 1M / 3M", f"{_pct(h.return_1w)} {_pct(h.return_1m)} {_pct(h.return_3m)}"),
            ("YTD / 1Y", f"{_pct(h.return_ytd)} {_pct(h.return_1y)}"),
            ("52w range", f"{_money(h.week52_low)} - {_money(h.week52_high)} "
                          f"(at {_plain_pct(h.week52_position)})"),
            ("50d / 200d avg", f"{_money(h.sma_50)} / {_money(h.sma_200)} "
                               f"({trend_label(h.price, h.sma_50, h.sma_200)})"),
            ("RSI 14", _num(h.rsi_14, ".0f")),
            ("Volatility 1Y", _plain_pct(h.volatility_1y)),
            ("Max drawdown 1Y", _plain_pct(h.max_drawdown_1y)),
            ("Beta (Yahoo / 1Y calc)", f"{_num(h.beta, '.2f')} / {_num(h.beta_1y, '.2f')}"),
        ],
        "Valuation & quality": [
            ("Market cap", _big(h.market_cap)),
            ("P/E (ttm / fwd)", f"{_num(h.pe_ratio)} / {_num(h.forward_pe)}"),
            ("PEG", _num(h.peg_ratio, ".2f")),
            ("P/S", _num(h.price_to_sales)),
            ("EV/EBITDA", _num(h.ev_to_ebitda)),
            ("Profit margin", _plain_pct(h.profit_margin)),
            ("Revenue / earnings growth",
             f"{_plain_pct(h.revenue_growth)} / {_plain_pct(h.earnings_growth)}"),
            ("Free cash flow", _big(h.free_cash_flow)),
            ("Debt/equity", _num(h.debt_to_equity, ".0f")),
        ],
        "Street & positioning": [
            ("Analyst rating", f"{h.analyst_rating or '-'} ({h.analyst_count or 0} analysts)"),
            ("Target low / mean / high", f"{_money(h.analyst_target_low)} / "
             f"{_money(h.analyst_target)} / {_money(h.analyst_target_high)}"),
            ("Upside to mean target", _pct(h.analyst_upside)),
            ("Short interest", f"{_plain_pct(h.short_pct_float, '.1%')} of float, "
                               f"{_num(h.short_ratio)} days to cover"),
            ("Institutions / insiders",
             f"{_plain_pct(h.inst_ownership)} / {_plain_pct(h.insider_ownership, '.1%')}"),
        ],
        "Events": [
            ("Next earnings", f"{h.next_earnings:%a %b %d, %Y}" if h.next_earnings else "-"),
            ("Last EPS surprise", _pct(h.last_eps_surprise_pct)),
            ("Dividend yield / payout",
             f"{_num(h.dividend_yield, '.2f')}% / {_plain_pct(h.payout_ratio)}"),
            ("Ex-dividend", f"{h.ex_dividend_date:%b %d, %Y}" if h.ex_dividend_date else "-"),
        ],
    }
    for title, rows in sections.items():
        table = Table(title=title, show_header=False, title_justify="left")
        table.add_column(style="dim")
        table.add_column(justify="right")
        for k, v in rows:
            table.add_row(k, v)
        console.print(table)

    lots = metrics.fifo_lots(snap.transactions).get(ticker, [])
    if lots:
        today = date.today()
        table = Table(title="Tax lots (FIFO)", title_justify="left")
        for col in ("Opened", "Shares", "Cost", "Gain", "Term", "Long-term on"):
            table.add_column(col, justify="right")
        for lot in lots:
            table.add_row(f"{lot.opened:%Y-%m-%d}", f"{lot.quantity:g}", _money(lot.price),
                          _money((h.price - lot.price) * lot.quantity),
                          "long" if lot.is_long_term(today) else "short",
                          f"{lot.long_term_date:%Y-%m-%d}")
        console.print(table)
    for _, held, in_lots in metrics.lot_mismatches([h], {ticker: lots}):
        console.print(f"[yellow]Lots cover {in_lots:g} of {held:g} shares; some history is "
                      f"missing, so treat these dates and gains as incomplete.[/yellow]")
    render_position_news(ticker, cfg)
    console.print(f"[dim]From snapshot taken {snap.taken_at:%Y-%m-%d %H:%M}[/dim]")


def render_position_news(ticker: str, cfg: dict) -> None:
    """Stored news for one ticker; `news` is what fetches and scores it."""
    from src.sentiment_pipeline import aggregate

    db = Path(cfg["database"])
    days = cfg.get("news", {}).get("lookback_days", 7)
    stored = storage.load_news([ticker], datetime.now(timezone.utc) - timedelta(days=days), db)
    if not stored:
        console.print(f"[dim]No stored news. Run `python -m src.cli news {ticker}`.[/dim]")
        return
    reads = aggregate.best_reads(stored, storage.load_sentiment([ticker], db_path=db))
    ts = aggregate.summarize(reads, ticker, cfg.get("sentiment", {}).get("half_life_hours", 48))
    console.print(f"\n[bold]News, last {days} days:[/bold] {ts.articles} articles, sentiment "
                  f"{_sent(ts.score)} ({ts.label}), {ts.positive} positive / "
                  f"{ts.negative} negative")
    for label, r in (("Most positive", ts.most_positive), ("Most negative", ts.most_negative)):
        if r:
            why = f" [dim]{r.rationale}[/dim]" if r.rationale else ""
            console.print(f"  {label} {_sent(r.score)}: {r.item.headline}{why}")
    for n in [n for n in stored if n.kind == "filing"][:5]:
        console.print(f"  [bold]{n.published_at:%b %d}[/bold] {n.headline}")


def _portfolio_tickers(cfg: dict) -> list[str]:
    """Holdings (largest first) plus option underlyings from the latest snapshot."""
    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    if snap is None:
        return []
    tickers = [h.ticker for h in sorted(snap.holdings, key=lambda h: -h.market_value)]
    return tickers + sorted({o.underlying for o in snap.options} - set(tickers))


def _ago(when: datetime) -> str:
    hours = (datetime.now(timezone.utc) - when).total_seconds() / 3600
    return f"{hours:.0f}h ago" if hours < 48 else f"{hours / 24:.0f}d ago"


def _sent(score: float | None) -> str:
    if score is None:
        return "-"
    color = "green" if score >= 0.15 else "red" if score <= -0.15 else "yellow"
    return f"[{color}]{score:+.2f}[/{color}]"


def cmd_news(args, cfg: dict) -> None:
    from src.data_pipeline.news import fetch_news
    from src.sentiment_pipeline import aggregate

    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    tickers = [t.upper() for t in args.tickers] or _portfolio_tickers(cfg)
    if not args.tickers and (cfg.get("watchlist") or {}).get("news", True):
        tickers += [t for t in _watchlist_tickers(cfg) if t not in tickers]
    if not tickers:
        console.print("No tickers given and no snapshot yet. Try `news AAPL` or run `snapshot`.")
        return
    news_cfg = {**cfg.get("news", {}), **({"lookback_days": args.days} if args.days else {})}
    sent_cfg = cfg.get("sentiment", {})
    db = Path(cfg["database"])
    names = {h.ticker: h.name for h in snap.holdings if h.name} if snap else {}
    if not args.no_fetch:
        with console.status(f"Fetching news for {len(tickers)} tickers..."):
            items = fetch_news(tickers, news_cfg, names=names)
            added = storage.save_news(items, db)
        console.print(f"Fetched {len(items)} items, {added} new.")

    since = datetime.now(timezone.utc) - timedelta(days=news_cfg.get("lookback_days", 7))
    stored = storage.load_news(tickers, since, db)
    if not args.no_score:
        with console.status("Scoring sentiment..."):
            counts = aggregate.score_new(stored, sent_cfg, db, names)
        if counts:
            console.print("Scored: " + ", ".join(f"{m} {n}" for m, n in counts.items()))
    reads = aggregate.best_reads(stored, storage.load_sentiment(tickers, db_path=db))
    read_by_key = {(r.item.ticker, r.item.url): r for r in reads}
    half_life = sent_cfg.get("half_life_hours", 48)

    summary = Table(title=f"News sentiment, last {news_cfg.get('lookback_days', 7)} days")
    for col in ("Ticker", "Score", "Read", "Articles", "+ / -", "Filings"):
        summary.add_column(col, justify="left" if col == "Ticker" else "right")
    for ticker in tickers:
        ts = aggregate.summarize(reads, ticker, half_life)
        filings = sum(1 for n in stored if n.ticker == ticker and n.kind == "filing")
        if ts.articles or filings:
            summary.add_row(ticker, _sent(ts.score), ts.label, str(ts.articles),
                            f"{ts.positive} / {ts.negative}", str(filings))
    console.print(summary)

    for ticker in tickers:
        mine = [n for n in stored if n.ticker == ticker]
        if not mine:
            continue
        table = Table(title=f"{ticker} ({len(mine)})", title_justify="left")
        for col in ("When", "Source", "Sent.", "Headline"):
            table.add_column(col, overflow="fold", justify="right" if col == "Sent." else "left")
        for n in mine[:args.limit]:
            read = read_by_key.get((n.ticker, n.url))
            head = f"[bold]{n.headline}[/bold]" if n.kind == "filing" else n.headline
            if read and read.rationale and args.why:
                head += f"\n[dim]{read.rationale}[/dim]"
            table.add_row(_ago(n.published_at), n.publisher or n.source,
                          _sent(read.score) if read else "-", head)
        console.print(table)


def cmd_options(args, cfg: dict) -> None:
    from statistics import mean

    from src.data_pipeline import options_data
    from src.portfolio_analysis import options as op

    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    tickers = [t.upper() for t in args.tickers] or _portfolio_tickers(cfg)
    if not tickers:
        console.print("No tickers given and no snapshot yet. Try `options AAPL` or run `snapshot`.")
        return
    held = {h.ticker: h for h in snap.holdings} if snap else {}
    db = Path(cfg["database"])
    today = date.today()
    with console.status("Fetching price history..."):
        closes = price_history(tickers, period="3y")

    table = Table(title="Options-implied expectations")
    for col in ("Ticker", "Price", "IV 30d", "Realized 30d", "IV rank", "~1W move",
                "~1M move", "Earnings", "Implied", "Past avg"):
        table.add_column(col, justify="left" if col == "Ticker" else "right")
    for ticker in tickers:
        with console.status(f"Option chains for {ticker}..."):
            h = held.get(ticker)
            earn = (h.next_earnings if h else None) or options_data.next_earnings(ticker)
            views = options_data.option_views(ticker, must_cover=[earn + timedelta(days=1)]
                                              if earn else None)
            if not views:
                table.add_row(ticker, *["-"] * 9)
                continue
            iv30 = op.interpolated_iv(views, 30)
            spot = views[0].spot
            if iv30:
                storage.save_iv(ticker, today, iv30, spot, db)
            history = storage.load_iv_history(ticker, today - timedelta(days=365), db)
            series = closes[ticker] if ticker in closes else None
            rv = op.realized_vol(series) if series is not None else None
            week = op.first_expiry_after(views, today + timedelta(days=5))
            month = op.first_expiry_after(views, today + timedelta(days=28))
            implied = op.earnings_move(views, earn) if earn else None
            past = []
            if earn and series is not None:
                past = op.past_earnings_moves(series, options_data.past_earnings_dates(ticker))

        rank = op.iv_rank(iv30, history)
        table.add_row(
            ticker, _money(spot), _plain_pct(iv30), _plain_pct(rv),
            _plain_pct(rank) if rank is not None else f"[dim]{len(history)}d[/dim]",
            f"±{week.expected_move_pct:.1%}" if week and week.expected_move_pct else "-",
            f"±{month.expected_move_pct:.1%}" if month and month.expected_move_pct else "-",
            f"{earn:%b %d}" if earn else "-",
            f"±{implied:.1%}" if implied else "-",
            f"±{mean(past):.1%} ({len(past)})" if past else "-",
        )
    console.print(table)
    console.print("[dim]Moves are the at-the-money straddle price. IV rank needs 20+ days of "
                  "stored readings (shown as Nd until then); run this daily to build it. "
                  "Past avg is the close-to-close move across recent reports.[/dim]")


def cmd_catalysts(args, cfg: dict) -> None:
    from src.data_pipeline import earnings_calendar, options_data
    from src.portfolio_analysis import catalysts
    from src.portfolio_analysis import options as op

    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return
    cat_cfg = cfg.get("catalysts", {})
    days = args.days or cat_cfg.get("horizon_days", 30)
    today = date.today()
    tickers = [h.ticker for h in snap.holdings if h.sector != "ETF"]
    with console.status("Checking earnings calendar..."):
        detail = earnings_calendar.fetch(tickers, today, today + timedelta(days=days))
    events = catalysts.build(snap, today, days, cat_cfg.get("macro_events"), detail)
    if not events:
        console.print(f"Nothing scheduled in the next {days} days.")
        return

    moves: dict[str, float | None] = {}
    if args.moves:
        for c in events:
            if c.kind == "earnings" and c.ticker not in moves:
                with console.status(f"Option chains for {c.ticker}..."):
                    views = options_data.option_views(
                        c.ticker, must_cover=[c.day + timedelta(days=1)])
                    moves[c.ticker] = op.earnings_move(views, c.day)

    table = Table(title=f"Catalysts, next {days} days (snapshot {snap.taken_at:%Y-%m-%d})")
    for col in ("Date", "In", "Ticker", "Weight", "Event", "Detail"):
        table.add_column(col, justify="right" if col in ("In", "Weight") else "left")
    for c in events:
        detail_text = c.detail
        if c.ticker in moves and c.kind == "earnings" and moves[c.ticker]:
            detail_text = ", ".join(filter(None, [detail_text,
                                                  f"options imply ±{moves[c.ticker]:.1%}"]))
        style = "bold" if c.kind == "earnings" and c.weight >= 0.05 else None
        table.add_row(f"{c.day:%a %b %d}", f"{(c.day - today).days}d", c.ticker or "market",
                      f"{c.weight:.0%}" if c.ticker else "", c.kind, detail_text, style=style)
    console.print(table)


def cmd_report(args, cfg: dict) -> list | None:
    from rich.markdown import Markdown

    from src.data_pipeline import earnings_calendar
    from src.portfolio_analysis import catalysts
    from src.sentiment_pipeline import aggregate
    from src.update_engine import alerts, daily, report, weekly
    from src.update_engine.changes import compare

    db = Path(cfg["database"])
    snap = storage.load_latest_snapshot(db)
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return None
    is_weekly = args.kind == "weekly"
    rep_cfg = cfg.get("reports", {})
    rep_cfg = {**rep_cfg, **rep_cfg.get("weekly", {})} if is_weekly else rep_cfg
    today = date.today()

    # Daily: the last run from an earlier day, so two runs today don't hide the move.
    # Weekly: the last run before this Monday, i.e. the end of last week.
    cutoff = (weekly.week_start(snap.taken_at.date()) if is_weekly
              else datetime.combine(snap.taken_at.date(), datetime.min.time()))
    prev = storage.load_snapshot_before(cutoff, snap.source, db)
    change = compare(prev, snap) if prev else None

    tickers = [h.ticker for h in sorted(snap.holdings, key=lambda h: -h.market_value)]
    news_hours = rep_cfg.get("news_hours", 36)
    since = datetime.now(timezone.utc) - timedelta(hours=news_hours)
    stored = storage.load_news(tickers, since, db)
    reads = aggregate.best_reads(stored, storage.load_sentiment(tickers, db_path=db))
    half_life = cfg.get("sentiment", {}).get("half_life_hours", 48)
    sentiment = [aggregate.summarize(reads, t, half_life) for t in tickers]

    horizon = rep_cfg.get("catalyst_days", 7)
    stocks = [h.ticker for h in snap.holdings if h.sector != "ETF"]
    with console.status("Checking earnings calendar..."):
        detail = earnings_calendar.fetch(stocks, today, today + timedelta(days=horizon))
    events = catalysts.build(snap, today, horizon,
                             cfg.get("catalysts", {}).get("macro_events"), detail)

    from src.update_engine import extras
    now = datetime.now(timezone.utc)
    stocks_by_value = extras.stock_tickers(snap)

    if args.kind == "alerts":
        new_trades = extras.new_insider_trades(stocks_by_value, cfg, db, now)
        new_calls = [(r["ticker"], r["quarter"], r["summary"]) for r in
                     storage.load_earnings_calls(stocks_by_value, now - timedelta(hours=24), db)]
        found = alerts.check(snap, cfg.get("alerts"), today, sentiment, events, stored,
                             insider_trades=new_trades, calls=new_calls)
        path = report.write_report("alerts", alerts.render(found, today), today)
        for a in found:
            color = "red" if a.level == "high" else "yellow"
            console.print(f"[{color}]{a.level:>5}[/{color}]  {a.message}")
        console.print(f"{len(found) or 'No'} alerts. Saved {path}")
        return found

    flags = metrics.risk_flags(snap, cfg["risk"], today)
    extra_sections = [
        extras.calls_section(stocks_by_value, db, now, days=7 if is_weekly else 3),
        extras.insider_section(stocks_by_value, cfg, db, today,
                               days=rep_cfg.get("insider_days", 30 if is_weekly else 14)),
        extras.macro_section(db, today),
    ]
    if is_weekly:
        markdown = weekly.render(snap, today, change, sentiment, events, flags,
                                 movers=rep_cfg.get("movers", 10),
                                 news_days=round(news_hours / 24), extras=extra_sections)
    else:
        markdown = daily.render(snap, today, change, sentiment, events, flags,
                                movers=rep_cfg.get("movers", 5),
                                news_window=f"last {news_hours} hours", extras=extra_sections)
    path = report.write_report(args.kind, markdown, today)
    if args.print:
        console.print(Markdown(markdown))
    if not stored:
        console.print("[dim]No stored news in the window. Run `python -m src.cli news` "
                      "first to include sentiment.[/dim]")
    if prev is None:
        earlier = "last week" if is_weekly else "an earlier day"
        console.print(f"[dim]No snapshot from {earlier} yet, so moves are vs. the "
                      "previous close and trades aren't shown.[/dim]")
    console.print(f"Saved {path}")
    return None


def _day_summary(snap: PortfolioSnapshot) -> str:
    moved = [(h.price - h.previous_close) * h.quantity for h in snap.holdings if h.previous_close]
    line = f"Equity {_money(snap.total_equity)}"
    if moved:
        today = sum(moved)
        base = snap.total_equity - today
        pct = f" ({today / base:+.2%})" if base else ""
        line += f", today {'+' if today >= 0 else '-'}${abs(today):,.0f}{pct}"
    return line


def _online(host: str = "api.robinhood.com") -> bool:
    import socket
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _wait_for_network(minutes: float, check=_online, sleep=None) -> bool:
    """A scheduled run often starts as the PC wakes, before Wi-Fi is back. Wait for it."""
    import time
    sleep = sleep or time.sleep
    deadline = time.monotonic() + minutes * 60
    while not check():
        if time.monotonic() >= deadline:
            return False
        console.print("Waiting for the network...")
        sleep(15)
    return True


def cmd_run(args, cfg: dict) -> None:
    """Snapshot, news, reports, alerts and ideas in one go, then notify. Built to run unattended:
    a step that fails is logged and reported, and the rest carry on with stored data."""
    import traceback
    from argparse import Namespace

    from src.update_engine import notify

    run_cfg = cfg.get("run", {})
    today = date.today()
    if today.weekday() >= 5 and not args.force:
        console.print(f"{today:%A}: markets are closed, nothing to do (use --force to run anyway).")
        return
    console.print(f"[bold]Run started {datetime.now():%Y-%m-%d %H:%M}[/bold]")
    wait = run_cfg.get("network_wait_min", 10)
    if not _wait_for_network(wait):
        console.print(f"[red]No internet after {wait} minutes;[/red] running on stored data.")

    failures: list[str] = []
    found: list = []

    def step(name: str, fn, ns: Namespace):
        console.rule(name)
        try:
            return fn(ns, cfg)
        except BaseException as e:  # noqa: BLE001 - includes SystemExit/EOFError from login prompts
            if isinstance(e, KeyboardInterrupt):
                raise
            console.print(f"[red]{name} failed:[/red] {e!r}")
            console.print(traceback.format_exc(), markup=False, highlight=False)
            failures.append(f"{name} failed: {e or type(e).__name__}")
            return None

    if not args.no_snapshot:
        step("snapshot", cmd_snapshot, Namespace(csv=None, no_save=False))
    if not args.no_news:
        step("news", cmd_news, Namespace(tickers=[], days=None, limit=3, no_fetch=False,
                                         no_score=False, why=False))
    if not args.no_news:
        step("alpha vantage", cmd_refresh_av, Namespace())
    step("daily report", cmd_report, Namespace(kind="daily", print=False))
    weekly_day = run_cfg.get("weekly_on", "friday").lower()
    if f"{today:%A}".lower() == weekly_day or args.weekly:
        step("weekly report", cmd_report, Namespace(kind="weekly", print=False))
    found = step("alerts", cmd_report, Namespace(kind="alerts", print=False)) or []
    idea_line = ""
    if not getattr(args, "no_ideas", False):
        result = step("ideas", cmd_ideas, Namespace(ticker=[], print=False))
        if result:
            from src.portfolio_analysis import ideas
            idea_line = ideas.headline(*result)

    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    summary = _day_summary(snap) if snap and snap.taken_at.date() == today else ""
    msg = notify.compose(found, failures, cfg.get("notify"), summary, idea_line)
    console.rule("notify")
    if msg is None or args.no_notify:
        console.print("Nothing to send." if msg is None else "Skipped (--no-notify).")
    else:
        for channel, result in notify.send(msg, cfg.get("notify")).items():
            console.print(f"{channel}: {result}")
    console.print(f"[bold]Run finished {datetime.now():%H:%M}[/bold], "
                  f"{len(failures)} step(s) failed.")
    if failures:
        raise SystemExit(1)


def cmd_refresh_av(args, cfg: dict) -> None:
    """Insider trades, macro series and new earnings call transcripts from Alpha Vantage."""
    from src.update_engine import extras

    db = Path(cfg["database"])
    snap = storage.load_latest_snapshot(db)
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return
    with console.status("Checking Alpha Vantage (insiders, macro, earnings calls)..."):
        res = extras.refresh(snap, cfg, db)
    calls = ", ".join(f"{t} {q}" for t, q in res.calls_new) or "none"
    console.print(f"Alpha Vantage: {res.insider_trades_new} new insider trades, "
                  f"macro updated: {', '.join(res.macro_updated) or 'already current'}, "
                  f"new earnings calls: {calls} ({res.calls_used} calls used)")
    if res.stopped:
        console.print(f"[yellow]Stopped early:[/yellow] {res.stopped}")


def cmd_insiders(args, cfg: dict) -> None:
    from src.portfolio_analysis import insiders
    from src.update_engine import extras

    db = Path(cfg["database"])
    snap = storage.load_latest_snapshot(db)
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return
    if not args.no_fetch:
        cmd_refresh_av(args, cfg)
    tickers = [t.upper() for t in args.tickers] or extras.stock_tickers(snap)
    trades = storage.load_insider_trades(tickers, date.today() - timedelta(days=args.days),
                                         db_path=db)
    picked = insiders.combine(trades) if args.all else insiders.notable(trades, cfg.get("insiders"))
    table = Table(title=f"Insider trades, last {args.days} days"
                        + ("" if args.all else " (open-market buys and large sales)"))
    for col in ("Date", "Ticker", "Insider", "Title", "Trade", "Shares", "Value"):
        table.add_column(col, justify="right" if col in ("Shares", "Value") else "left",
                         overflow="fold")
    for t in sorted(picked, key=lambda t: t.day, reverse=True)[:args.limit]:
        color = "green" if t.code == "P" else "red" if t.code == "S" else "dim"
        table.add_row(f"{t.day:%b %d}", t.ticker, t.name, t.title,
                      f"[{color}]{t.verb or t.code}[/{color}]", f"{t.shares:,.0f}",
                      insiders.money(t.value) if t.value else "-")
    console.print(table)
    if not picked:
        console.print("[dim]Nothing notable. `--all` shows every filing, including option "
                      "exercises and tax withholding.[/dim]")


def cmd_macro(args, cfg: dict) -> None:
    from rich.markdown import Markdown

    from src.update_engine import extras

    db = Path(cfg["database"])
    if not args.no_fetch and storage.load_latest_snapshot(db) is not None:
        cmd_refresh_av(args, cfg)
    section = extras.macro_section(db, date.today())
    console.print(Markdown("\n".join(section)) if section else
                  "No macro data yet. Set ALPHAVANTAGE_API_KEY and run without --no-fetch.")


def cmd_earnings_call(args, cfg: dict) -> None:
    import json

    from rich.markdown import Markdown

    from src.data_pipeline import alpha_vantage as av
    from src.update_engine import extras

    db = Path(cfg["database"])
    ticker = args.ticker.upper()
    rows = [r for r in storage.load_earnings_calls([ticker], db_path=db)
            if not args.quarter or r["quarter"] == args.quarter]
    if not rows or args.refresh:
        # Default: this quarter's label (companies whose fiscal year runs ahead), then the
        # last one, which most companies have reported by now. Up to two Alpha Vantage calls.
        quarters = [args.quarter] if args.quarter else av.recent_quarters(date.today())
        client = av.Client(len(quarters))
        for quarter in quarters:
            with console.status(f"Fetching the {ticker} {quarter} call..."):
                entries = av.transcript(client, ticker, quarter)
            if entries:
                summary = extras._summarize(ticker, quarter, entries, cfg)
                storage.save_earnings_call(ticker, quarter, None, json.dumps(entries), summary,
                                           (cfg.get("earnings_calls") or {}).get("model")
                                           if summary else None,
                                           datetime.now(timezone.utc), db)
                break
        rows = [r for r in storage.load_earnings_calls([ticker], db_path=db)
                if not args.quarter or r["quarter"] == args.quarter]
    if not rows:
        console.print(f"No transcript found for {ticker}"
                      + (f" {args.quarter}" if args.quarter else "") + ".")
        return
    r = rows[0]
    console.print(Markdown(f"**{ticker} {r['quarter']} earnings call**\n\n"
                           + (r["summary"] or "_No summary: set ANTHROPIC_API_KEY._")))


def _signal_inputs(tickers: list[str], cfg: dict, today: date) -> tuple[list, list]:
    """Stored news sentiment and open-market insider buys for these tickers."""
    from src.portfolio_analysis.insiders import combine
    from src.sentiment_pipeline import aggregate

    db = Path(cfg["database"])
    idea_cfg = _idea_cfg(cfg)
    since = datetime.now(timezone.utc) - timedelta(hours=idea_cfg.get("news_hours", 72))
    reads = aggregate.best_reads(storage.load_news(tickers, since, db),
                                 storage.load_sentiment(tickers, db_path=db))
    half_life = cfg.get("sentiment", {}).get("half_life_hours", 48)
    sentiment = [aggregate.summarize(reads, t, half_life) for t in tickers]
    buys = combine(storage.load_insider_trades(
        tickers, today - timedelta(days=idea_cfg.get("insider_days", 90)), db_path=db))
    return sentiment, buys


def _ideas(snap: PortfolioSnapshot, cfg: dict, today: date) -> list:
    """Ideas for the snapshot's holdings from stored news sentiment and insider buys."""
    from src.portfolio_analysis import ideas
    from src.update_engine import extras

    sentiment, _ = _signal_inputs([h.ticker for h in snap.holdings], cfg, today)
    _, buys = _signal_inputs(extras.stock_tickers(snap), cfg, today)
    return ideas.suggest(snap, _idea_cfg(cfg), today, sentiment, buys)


def _watchlist_tickers(cfg: dict) -> list[str]:
    return [t.upper() for t in (cfg.get("watchlist") or {}).get("tickers") or []]


def _candidates(tickers: list[str], cfg: dict) -> list[Holding]:
    """Tickers you don't own, filled in like holdings so they can be scored the same way."""
    if not tickers:
        return []
    found = [Holding(t, 0, 0.0, 0.0) for t in tickers]
    enrich_holdings(found)
    bench = cfg["benchmarks"][0]
    apply_price_stats(found, price_history(tickers + [bench]), bench)
    return found


def _watch_ideas(tickers: list[str], snap: PortfolioSnapshot, cfg: dict,
                 today: date) -> tuple[list, list[Holding]]:
    """Watchlist ideas, plus the looked-up tickers they came from."""
    from src.portfolio_analysis import ideas

    held = {h.ticker for h in snap.holdings}
    tickers = [t for t in dict.fromkeys(tickers) if t not in held]
    if not tickers:
        return [], []
    with console.status(f"Looking up {len(tickers)} watchlist tickers..."):
        candidates = _candidates(tickers, cfg)
    sentiment, buys = _signal_inputs(tickers, cfg, today)
    return ideas.watchlist(candidates, snap, _idea_cfg(cfg), today, sentiment, buys), candidates


def _idea_cfg(cfg: dict) -> dict:
    return {"max_weight": cfg["risk"]["max_position_weight"], **cfg.get("ideas", {})}


def _options_section(found: list, watch: list, candidates: list[Holding],
                     snap: PortfolioSnapshot, cash: float, cfg: dict, today: date) -> str:
    """Covered calls and cash-secured puts for the day's ideas, as a report section."""
    from src.data_pipeline import options_data
    from src.portfolio_analysis import option_ideas
    from src.portfolio_analysis.options import expiry_view, iv_rank

    db = Path(cfg["database"])
    opt_cfg = cfg.get("option_ideas") or {}
    ranks: dict[str, float] = {}

    def chain_for(ticker: str, earnings: date | None):
        chain = options_data.chain_near(ticker, earnings, opt_cfg, today)
        if chain is not None:
            view = expiry_view(chain.calls, chain.puts, chain.spot, chain.expiration, today)
            history = storage.load_iv_history(ticker, today - timedelta(days=365), db)
            rank = iv_rank(view.atm_iv if view else None, history)
            if rank is not None:
                ranks[ticker] = rank
        return chain

    with console.status("Option chains for covered calls and puts..."):
        calls = option_ideas.covered_calls(snap, found, chain_for, opt_cfg, today, ranks)
        info = {h.ticker: h for h in snap.holdings + candidates}
        puts, too_big = option_ideas.cash_secured_puts(
            snap, found + watch, info, cash, chain_for, opt_cfg, today, ranks,
            cfg["risk"]["max_position_weight"])
    return option_ideas.render(calls, puts, too_big)


def _odds_section(cfg: dict, today: date) -> str:
    """Prediction market odds for the series in settings, stored so the next run can show
    how they moved."""
    from src.data_pipeline import kalshi
    from src.portfolio_analysis import odds

    c = cfg.get("odds") or {}
    db = Path(cfg["database"])
    events = []
    with console.status("Prediction market odds from Kalshi..."):
        for series in c.get("series") or []:
            events += kalshi.fetch_series(series, c.get("events", 2))
    if events:
        storage.save_odds(events, today, db)
    back = today - timedelta(days=c.get("compare_days", 7))
    before = {e.event: b for e in events if (b := storage.load_odds_before(e.event, back, db))}
    return odds.render(events, before, today)


def cmd_odds(args, cfg: dict) -> None:
    from rich.markdown import Markdown

    console.print(Markdown(_odds_section(cfg, date.today())))


def _safe_section(name: str, build) -> str:
    """A report section that can't take the rest of the report down with it."""
    try:
        return build()
    except Exception as e:  # noqa: BLE001 - network and data errors from third parties
        console.print(f"[red]{name} failed:[/red] {e!r}")
        return f"## {name}\n\n_Couldn't build this section: {e}_\n"


def cmd_ideas(args, cfg: dict):
    """Print the ideas and, for the whole portfolio, save them with a cash plan.
    Returns (holding and watchlist ideas, cash plan) so `run` can put them in the
    notification.

    With tickers, held ones are filtered and the rest are scored like watchlist tickers."""
    from rich.markdown import Markdown

    from src.portfolio_analysis import ideas
    from src.update_engine import report

    snap = storage.load_latest_snapshot(Path(cfg["database"]))
    if snap is None:
        console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
        return None
    today = date.today()
    found = _ideas(snap, cfg, today)
    if args.ticker:
        wanted = [t.upper() for t in args.ticker]
        found = [i for i in found if i.ticker in wanted]
    watch, candidates = _watch_ideas(wanted if args.ticker else _watchlist_tickers(cfg),
                                     snap, cfg, today)
    plan = ideas.deploy_cash(found + watch, snap, _idea_cfg(cfg))
    sections = []
    if not getattr(args, "no_options", False):
        sections.append(_safe_section("Options", lambda: _options_section(
            found, watch, candidates, snap, plan.investable, cfg, today)))
    if not args.ticker and not getattr(args, "no_odds", False):
        sections.append(_safe_section("Market odds", lambda: _odds_section(cfg, today)))
    markdown = ideas.render(found, snap, today, None if args.ticker else plan,
                            watch or (None if args.ticker else []), sections)
    if getattr(args, "print", True):
        console.print(Markdown(markdown))
    if not args.ticker:
        console.print(f"Saved {report.write_report('ideas', markdown, today)}")
    return found + watch, None if args.ticker else plan


def cmd_notify(args, cfg: dict) -> None:
    from src.update_engine import alerts, notify

    if args.test:
        fake = [alerts.Alert("high", "TEST", "move", "TEST up 6.0% today (this is a test)")]
        msg = notify.compose(fake, [], cfg.get("notify"), "Test notification from the analyzer")
    else:
        snap = storage.load_latest_snapshot(Path(cfg["database"]))
        if snap is None:
            console.print("No snapshots yet. Run `python -m src.cli snapshot` first.")
            return
        found = alerts.check(snap, cfg.get("alerts"))
        msg = notify.compose(found, [], {**cfg.get("notify", {}), "always": True},
                             _day_summary(snap))
    console.print(f"[bold]{msg.title}[/bold]\n{msg.body}\n")
    for channel, result in notify.send(msg, cfg.get("notify")).items():
        console.print(f"{channel}: {result}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="portfolio")
    parser.add_argument("--config", default="configs/settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="pull current portfolio and save a snapshot")
    snap.add_argument("--csv", help="Robinhood activity CSV instead of logging in")
    snap.add_argument("--no-save", action="store_true")
    snap.set_defaults(func=cmd_snapshot)

    hist = sub.add_parser("history", help="list saved snapshots")
    hist.add_argument("--limit", type=int, default=30)
    hist.set_defaults(func=cmd_history)

    pos = sub.add_parser("position", help="deep dive on one holding (latest snapshot)")
    pos.add_argument("ticker")
    pos.set_defaults(func=cmd_position)

    news = sub.add_parser("news", help="fetch and show news and SEC filings for holdings")
    news.add_argument("tickers", nargs="*", help="defaults to everything in the latest snapshot")
    news.add_argument("--days", type=int, help="lookback window (default from settings)")
    news.add_argument("--limit", type=int, default=10, help="rows shown per ticker")
    news.add_argument("--no-fetch", action="store_true", help="only show what's stored")
    news.add_argument("--no-score", action="store_true", help="skip sentiment scoring")
    news.add_argument("--why", action="store_true", help="show the LLM's reason under each headline")
    news.set_defaults(func=cmd_news)

    opts = sub.add_parser("options", help="implied moves, IV and earnings expectations")
    opts.add_argument("tickers", nargs="*", help="defaults to everything in the latest snapshot")
    opts.set_defaults(func=cmd_options)

    cats = sub.add_parser("catalysts", help="upcoming earnings, dividends, expiries and more")
    cats.add_argument("--days", type=int, help="how far ahead to look (default from settings)")
    cats.add_argument("--moves", action="store_true",
                      help="add the options-implied earnings move (slower: fetches chains)")
    cats.set_defaults(func=cmd_catalysts)

    rep = sub.add_parser("report", help="write a report to reports/ (stored data, no news fetch)")
    rep.add_argument("kind", choices=["daily", "weekly", "alerts"])
    rep.add_argument("--print", action="store_true", help="also show it in the terminal")
    rep.set_defaults(func=cmd_report)

    ins = sub.add_parser("insiders", help="insider buying and selling in your holdings")
    ins.add_argument("tickers", nargs="*", help="defaults to everything in the latest snapshot")
    ins.add_argument("--days", type=int, default=90)
    ins.add_argument("--limit", type=int, default=30)
    ins.add_argument("--all", action="store_true", help="every filing, not just notable trades")
    ins.add_argument("--no-fetch", action="store_true", help="only show what's stored")
    ins.set_defaults(func=cmd_insiders)

    mac = sub.add_parser("macro", help="Fed funds, inflation and the 10-year yield")
    mac.add_argument("--no-fetch", action="store_true", help="only show what's stored")
    mac.set_defaults(func=cmd_macro)

    call = sub.add_parser("earnings-call", help="summary of a holding's latest earnings call")
    call.add_argument("ticker")
    call.add_argument("--quarter", help="e.g. 2026Q3 (default: the latest one available)")
    call.add_argument("--refresh", action="store_true", help="fetch again even if stored")
    call.set_defaults(func=cmd_earnings_call)

    idea = sub.add_parser("ideas", help="add / hold / trim / sell ideas for your holdings")
    idea.add_argument("ticker", nargs="*", help="only show these (default: all, and save)")
    idea.add_argument("--no-options", action="store_true", help="skip covered calls and puts")
    idea.add_argument("--no-odds", action="store_true", help="skip prediction market odds")
    idea.set_defaults(func=cmd_ideas)

    odds = sub.add_parser("odds", help="prediction market odds for rates, inflation and more")
    odds.set_defaults(func=cmd_odds)

    run = sub.add_parser("run", help="snapshot, news, reports, alerts and ideas, then notify")
    run.add_argument("--force", action="store_true", help="run on weekends too")
    run.add_argument("--weekly", action="store_true", help="write the weekly report today")
    run.add_argument("--no-snapshot", action="store_true", help="skip the Robinhood pull")
    run.add_argument("--no-news", action="store_true", help="skip fetching news")
    run.add_argument("--no-ideas", action="store_true", help="skip the ideas report")
    run.add_argument("--no-notify", action="store_true", help="don't send notifications")
    run.set_defaults(func=cmd_run)

    note = sub.add_parser("notify", help="send the latest alerts to your notification channels")
    note.add_argument("--test", action="store_true", help="send a sample alert instead")
    note.set_defaults(func=cmd_notify)

    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.func(args, cfg)


if __name__ == "__main__":
    main()
