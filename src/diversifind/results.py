"""Structured result objects for diversification search outputs.

These dataclasses provide a cleaner and more user-friendly public API than
nested dictionaries. Search methods can construct and return these objects so
users can access portfolio results via attributes instead of string keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class PortfolioEntry:
    """Single ranked portfolio returned by a search method.

    Attributes:
        rank: Rank of the portfolio within the returned results.
        logdet: Log-determinant diversification score.
        combo_indices: Selected asset indices.
        combo_symbols: Selected asset symbols.
    """

    rank: int
    logdet: float
    combo_indices: List[int] = field(default_factory=list)
    combo_symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry into a JSON-serializable dictionary."""
        return {
            "rank": int(self.rank),
            "logdet": float(self.logdet),
            "combo_indices": [int(i) for i in self.combo_indices],
            "combo_symbols": [str(s) for s in self.combo_symbols],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PortfolioEntry":
        """Construct an entry from a dictionary payload."""
        return cls(
            rank=int(payload.get("rank", 0)),
            logdet=float(payload.get("logdet", float("nan"))),
            combo_indices=[int(i) for i in payload.get("combo_indices", [])],
            combo_symbols=[str(s) for s in payload.get("combo_symbols", [])],
        )


@dataclass(slots=True)
class PortfolioResult:
    """Structured search result returned by a diversification method.

    Attributes:
        method: Search method name.
        k: Target portfolio size.
        top_k: Number of requested results.
        top_results: Ranked portfolio entries.
        metadata: Additional method-specific metadata (e.g. beam width,
            worker count).
    """

    method: str
    k: int
    top_k: int
    top_results: List[PortfolioEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def best(self) -> PortfolioEntry:
        """Return the best portfolio entry.

        Raises:
            ValueError: If no portfolios are present.
        """
        if not self.top_results:
            raise ValueError("No portfolio results available")
        return self.top_results[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a JSON-serializable dictionary."""
        payload: Dict[str, Any] = {
            "method": str(self.method),
            "k": int(self.k),
            "top_k": int(self.top_k),
            "top_results": [entry.to_dict() for entry in self.top_results],
        }
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PortfolioResult":
        """Construct a result object from a dictionary payload."""
        reserved = {"method", "k", "top_k", "top_results"}
        metadata = {k: v for k, v in payload.items() if k not in reserved}
        return cls(
            method=str(payload.get("method", "unknown")),
            k=int(payload.get("k", 0)),
            top_k=int(payload.get("top_k", 0)),
            top_results=[PortfolioEntry.from_dict(x) for x in payload.get("top_results", [])],
            metadata=metadata,
        )

    def summary_lines(self) -> List[str]:
        """Return a compact human-readable summary as a list of lines."""
        lines = [
            f"Method: {self.method}",
            f"Portfolio size: {self.k}",
            f"Returned results: {len(self.top_results)}",
        ]
        if self.metadata:
            for key, value in self.metadata.items():
                lines.append(f"{key}: {value}")
        return lines

    def pretty(self) -> str:
        """Return a formatted string representation of the result."""
        lines = self.summary_lines()

        if self.top_results:
            lines.append("")
            lines.append("Top diversified portfolios:")
            lines.append("")

            header = f"{'Rank':>4}  {'LogDet':>10}  Symbols"
            separator = "-" * len(header)

            lines.append(header)
            lines.append(separator)

            for entry in self.top_results:
                symbols = ", ".join(entry.combo_symbols)
                lines.append(f"{entry.rank:>4}  {entry.logdet:>10.6f}  {symbols}")

        return "\n".join(lines)


def result_from_dict(payload: Dict[str, Any]) -> PortfolioResult:
    """Convenience wrapper for constructing a result from a dictionary."""
    return PortfolioResult.from_dict(payload)


__all__ = [
    "PortfolioEntry",
    "PortfolioResult",
    "result_from_dict",
]
