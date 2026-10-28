from __future__ import annotations

from abc import ABC, abstractmethod

from src.data_pipeline.models import PortfolioSnapshot


class PortfolioSource(ABC):
    """Anything that can produce a PortfolioSnapshot (Robinhood API, CSV export, ...)."""

    name: str

    @abstractmethod
    def fetch_snapshot(self) -> PortfolioSnapshot:
        """Return current holdings, account cash and transaction history."""
