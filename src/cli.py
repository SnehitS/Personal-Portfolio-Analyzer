"""Command line entry point.

    python -m src.cli snapshot                    # pull from Robinhood
    python -m src.cli snapshot --csv export.csv   # rebuild from a Robinhood activity CSV
    python -m src.cli history                     # list saved snapshots
    python -m src.cli position NVDA               # deep dive from the latest snapshot
"""

from __future__ import annotations

import argparse
from datetime import date
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
    console.print(f"[dim]From snapshot taken {snap.taken_at:%Y-%m-%d %H:%M}[/dim]")


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

    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.func(args, cfg)


if __name__ == "__main__":
    main()
