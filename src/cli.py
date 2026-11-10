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
        source = RobinhoodSource()

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


def cmd_report(args, cfg: dict) -> None:
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
        return
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

    if args.kind == "alerts":
        found = alerts.check(snap, cfg.get("alerts"), today, sentiment, events, stored)
        path = report.write_report("alerts", alerts.render(found, today), today)
        for a in found:
            color = "red" if a.level == "high" else "yellow"
            console.print(f"[{color}]{a.level:>5}[/{color}]  {a.message}")
        console.print(f"{len(found) or 'No'} alerts. Saved {path}")
        return

    flags = metrics.risk_flags(snap, cfg["risk"], today)
    if is_weekly:
        markdown = weekly.render(snap, today, change, sentiment, events, flags,
                                 movers=rep_cfg.get("movers", 10),
                                 news_days=round(news_hours / 24))
    else:
        markdown = daily.render(snap, today, change, sentiment, events, flags,
                                movers=rep_cfg.get("movers", 5),
                                news_window=f"last {news_hours} hours")
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

    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.func(args, cfg)


if __name__ == "__main__":
    main()
