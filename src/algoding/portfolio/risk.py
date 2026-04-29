from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from algoding.execution.models import OrderIntent


@dataclass
class RiskLimits:
    max_position_risk_bps: int = 50
    max_daily_loss_bps: int = 150
    max_gross_exposure: float = 1.0
    max_turnover_per_day: float = 0.25
    heartbeat_timeout_seconds: int = 30
    stale_data_threshold_seconds: int = 15
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.07
    max_live_positions: int = 5
    max_promoted_position_pct: float = 0.02

    @classmethod
    def from_file(cls, path: Path | str = Path("config/risk.yaml")) -> "RiskLimits":
        payload: dict[str, str] = {}
        file_path = Path(path)
        if file_path.exists():
            for raw_line in file_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                payload[key.strip()] = value.strip()
        return cls(
            max_position_risk_bps=int(payload.get("max_position_risk_bps", cls.max_position_risk_bps)),
            max_daily_loss_bps=int(payload.get("max_daily_loss_bps", cls.max_daily_loss_bps)),
            max_gross_exposure=float(payload.get("max_gross_exposure", cls.max_gross_exposure)),
            max_turnover_per_day=float(payload.get("max_turnover_per_day", cls.max_turnover_per_day)),
            heartbeat_timeout_seconds=int(
                payload.get("heartbeat_timeout_seconds", cls.heartbeat_timeout_seconds)
            ),
            stale_data_threshold_seconds=int(
                payload.get("stale_data_threshold_seconds", cls.stale_data_threshold_seconds)
            ),
            stop_loss_pct=float(payload.get("stop_loss_pct", cls.stop_loss_pct)),
            trailing_stop_pct=float(payload.get("trailing_stop_pct", cls.trailing_stop_pct)),
            max_live_positions=int(payload.get("max_live_positions", cls.max_live_positions)),
            max_promoted_position_pct=float(
                payload.get("max_promoted_position_pct", cls.max_promoted_position_pct)
            ),
        )

    def with_overrides(
        self,
        *,
        stop_loss_pct: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> "RiskLimits":
        return replace(
            self,
            stop_loss_pct=self.stop_loss_pct if stop_loss_pct is None else stop_loss_pct,
            trailing_stop_pct=self.trailing_stop_pct if trailing_stop_pct is None else trailing_stop_pct,
        )


@dataclass
class PositionRiskState:
    symbol: str
    qty: float
    avg_entry_price: float | None
    current_price: float | None
    market_value: float
    high_water_mark: float | None


@dataclass
class StopDecision:
    should_exit: bool
    reason: str | None
    fixed_stop_price: float | None
    trailing_stop_price: float | None
    effective_stop_price: float | None
    high_water_mark: float | None


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits.from_file()

    def validate(self, intent: OrderIntent) -> None:
        if intent.quantity > 0:
            return
        if intent.notional is not None and intent.notional > 0:
            return
        raise ValueError("Order must have a positive quantity or notional value.")

    def validate_promoted_entry(
        self,
        intent: OrderIntent,
        account_equity: float,
        gross_exposure_ratio: float,
        open_positions_count: int,
        current_position_qty: float,
    ) -> None:
        self.validate(intent)
        if intent.side.lower() != "buy":
            return
        if account_equity <= 0:
            raise ValueError("Account equity must be positive before entering a position.")
        if intent.notional is None or intent.notional <= 0:
            raise ValueError("Promoted buy orders must specify positive notional size.")

        position_ratio = intent.notional / account_equity
        if position_ratio > self._limits.max_promoted_position_pct:
            raise ValueError(
                f"Promoted position size {position_ratio:.2%} exceeds max_promoted_position_pct "
                f"{self._limits.max_promoted_position_pct:.2%}."
            )

        projected_gross = gross_exposure_ratio + position_ratio
        if projected_gross > self._limits.max_gross_exposure:
            raise ValueError(
                f"Projected gross exposure {projected_gross:.2f} exceeds limit {self._limits.max_gross_exposure:.2f}."
            )

        if current_position_qty <= 0 and open_positions_count >= self._limits.max_live_positions:
            raise ValueError(
                f"Open positions {open_positions_count} already at max_live_positions {self._limits.max_live_positions}."
            )

    def check_daily_loss_halt(self, account_equity: float, reference_equity: float | None) -> None:
        if reference_equity is None or reference_equity <= 0 or account_equity <= 0:
            return
        drawdown_bps = ((reference_equity - account_equity) / reference_equity) * 10_000
        if drawdown_bps > self._limits.max_daily_loss_bps:
            raise ValueError(
                f"Daily loss halt triggered at {drawdown_bps:.0f} bps, above limit {self._limits.max_daily_loss_bps} bps."
            )

    def evaluate_long_stops(self, state: PositionRiskState) -> StopDecision:
        if state.qty <= 0 or state.current_price is None:
            return StopDecision(False, None, None, None, None, state.high_water_mark)

        high_water = max(
            [value for value in [state.high_water_mark, state.avg_entry_price, state.current_price] if value is not None]
        )
        fixed_stop = (
            state.avg_entry_price * (1 - self._limits.stop_loss_pct)
            if state.avg_entry_price is not None
            else None
        )
        trailing_stop = high_water * (1 - self._limits.trailing_stop_pct)
        effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)

        if fixed_stop is not None and state.current_price <= fixed_stop:
            return StopDecision(True, "stop_loss", fixed_stop, trailing_stop, effective_stop, high_water)
        if state.current_price <= trailing_stop:
            return StopDecision(True, "trailing_stop", fixed_stop, trailing_stop, effective_stop, high_water)
        return StopDecision(False, None, fixed_stop, trailing_stop, effective_stop, high_water)

    @staticmethod
    def is_snapshot_from_today(created_at: str) -> bool:
        try:
            snapshot_time = datetime.fromisoformat(created_at)
        except ValueError:
            return False
        now = datetime.now(timezone.utc).date()
        return snapshot_time.astimezone(timezone.utc).date() == now
