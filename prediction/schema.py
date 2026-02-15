"""
Prediction output schema.

Every prediction the platform produces gets wrapped in this dataclass.
Keeps the output contract explicit so downstream consumers (dashboard, alerts,
backtest) always know what to expect.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import json


@dataclass
class Prediction:
    symbol: str                                # e.g. "BTC/USD"
    timestamp: str = ""                        # ISO-8601, filled automatically if empty
    action: str = "HOLD"                       # "BUY" | "SELL" | "HOLD"
    confidence: float = 0.0                    # 0.0 to 1.0
    horizon: str = "1h"                        # "1h" | "4h" | "1d"
    factor_breakdown: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""                        # Human-readable explanation
    risk_flags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0                    # End-to-end prediction time
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Normalize action
        self.action = self.action.upper()
        if self.action not in ("BUY", "SELL", "HOLD"):
            self.action = "HOLD"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def summary(self) -> str:
        """One-liner for logging / alerts."""
        flags = ", ".join(self.risk_flags) if self.risk_flags else "none"
        return (
            f"[{self.symbol}] {self.action} "
            f"(conf={self.confidence:.0%}, horizon={self.horizon}) "
            f"risk_flags=[{flags}] latency={self.latency_ms:.0f}ms"
        )
