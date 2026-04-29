from __future__ import annotations


def rank_by_simple_momentum(
    symbol_closes: dict[str, list[float]], lookback_bars: int
) -> list[dict[str, float | str]]:
    rankings: list[dict[str, float | str]] = []
    for symbol, closes in symbol_closes.items():
        if len(closes) < lookback_bars or closes[0] <= 0:
            continue
        momentum = (closes[-1] / closes[0]) - 1
        rankings.append(
            {
                "symbol": symbol,
                "start_close": closes[0],
                "end_close": closes[-1],
                "momentum": momentum,
            }
        )
    rankings.sort(key=lambda item: float(item["momentum"]), reverse=True)
    return rankings
