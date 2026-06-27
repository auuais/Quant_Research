# V3-5 review and V3-Q3E design

This document is Part 1 (critical review of the existing V3 / V3-5 directional pipeline) and the
design notes for Part 2 (the new V3-Q3E branch built on `Qwen/Qwen3-Embedding-4B`).

All claims below were checked against the code and the committed report artifacts.

## Part 1 — review of V3-5

V3-5 = `run_v3_top5_trailing_no_same_day_reentry`
(`src/algoding/execution/deepseek_directional_v3_variants.py`) replaying the DeepSeek V3
finetuned directional scores under a 5% stop + 1% trailing stop + no-same-day-reentry policy.

### 1. Is V3-5 execution-aware? **No.**
- V3-5 reports `build_directional_trade_trace(...)["summary"]["total_return"]` directly
  (`deepseek_directional_v3_variants.py:103`). That function is a **gross** simulator: no spread,
  no slippage, no fees. With 63–126 round trips per symbol, costs are not negligible.
- The parent V3 run (`deepseek_directional_research.py:211-212`) does call
  `run_execution_aware_replay`, but the trace emits exit reasons `"risk_exit"` / `"prediction_exit"`,
  while the replay only adds stop slippage / intrabar-low flooring for reasons in
  `{"stop_loss","trailing_stop"}` (`cost_aware_backtest.py:84-93`). So even there the stop-specific
  realism is silently skipped — a reason-string mismatch bug.

### 2. Do the labels leak future information? **Labels are correct; the entry timing is optimistic; boundaries are not purged.**
- The 5-day forward return is the label and *should* look forward — that is not leakage
  (`news_directional_labeler.py:54-56`).
- Features and entry both use `close[T]`, and the label is measured from `close[T]`. There is no
  `close[T+k]` in the features, so there is **no hard future-price leak**.
- News bundling assigns articles before the 16:00 ET close to day `T` (`news_labeler.py:301-310`),
  and entry is at `close[T]`. That is internally consistent but **optimistic**: it assumes you can
  read intraday news up to 15:59 and fill exactly at the official close.

### 3. Are the splits strict? **Chronological, but no purge/embargo.**
- Splits are a clean chronological 60/20/20 with no shuffling (`deepseek_directional_research.py:292-306`). Good.
- But there is **no purge/embargo**. A train sample at day `T` has a label that ends at `close[T+5]`;
  for the last ~5 train days that `T+5` falls in the validation period, so those labels read
  validation prices. Same at the validation→test seam. Small in magnitude, real in principle.

### 4. Are stop / trailing / no-same-day-reentry / slippage applied consistently? **Stops yes, slippage no.**
- Stop, trailing stop, and no-same-day-reentry are implemented and applied
  (`deepseek_directional_model.py:393-458`). Slippage/fees are **not** applied in V3-5 (point 1).

### 5. Are V3-5's reported results from the same benchmark window? **No — verified mismatch.**
- Per-symbol accuracy is **copied** from `deepseek_directional_run.json`
  (test window `2025-02-24 → 2026-04-06`, `deepseek_directional_v3_variants.py:32-38`).
- Per-symbol returns are **recomputed** on a fresh `get_recent_bars` reload whose test window is
  `2025-02-25 → 2026-04-07` (see `v3_5_top5_trailing_1pct_no_same_day_reentry.json`). Because
  `get_recent_bars` is always relative to `now()` (`historical.py:63-70`), running a day later shifts
  the window by one trading day. **Accuracy and returns therefore come from different windows.**
- The dated artifacts (`v3_5_top5_20260326_20260626`, `v3_5_top5_20250626_20260626`) report returns
  only — no accuracy — so they cannot be cross-audited at all.

### 6. Are predictions used for entry, exit, or both? **Both, at the close — but exits are stop-driven in practice.**
- Entry requires `direction == bullish` and `strength ≥ medium`; exit fires when the signal is no
  longer long *or* a stop triggers (`deepseek_directional_model.py:420,448`).
- In the committed run, exits are essentially 100% `risk_exit`: AVGO 126/126, NVDA 85/85, TSLA 101/101,
  GOOGL 63/63, XOM 115/116. **The model's exit signal almost never fires — the 1% trailing stop does.**

### 7. Is momentum/rule logic delaying entries? **Not in V3-5's reported config; yes in its siblings.**
- V3-5 sets `require_momentum_confirmation=False`, so momentum does not gate entries there.
- But `select_best_directional_trade_definition` sweeps momentum on/off, and
  `build_directional_trade_trace_with_momentum_shift_entry` requires a fresh 0→1 momentum cross to
  enter (`deepseek_directional_model.py:621`), which **does** delay sentiment-driven entries. Keep
  momentum off the primary V3-Q3E run (as the task requests).

### 8. Is the sentiment model overfit to recent data? **The LM is trained cleanly; the trade mapper and caching are the bigger risks.**
- The LoRA fine-tune uses chronological train only (good). Scoring is greedy/deterministic.
- The headline V3-5 gating (medium / no-rel / no-mom / 1% trailing) looks hand-picked rather than
  validation-selected, and the directional score cache is reused across windows and is **stale**
  (`finetuned_directional_scores.jsonl` covers the 5 eval symbols only through **2026-04-10**).

### 9. Are the artifacts enough to audit? **Partially.**
- The parent V3 JSON keeps `trace_points` + `events`. The V3-5 variant JSON keeps only summary rows
  plus the copied accuracy — no per-trade events, no equity curve, no confusion matrix on the return
  window. So you cannot reconcile "30.6% directional accuracy" with "+645% return" from the artifacts.

### Bottom line
A 30.6% three-class directional accuracy producing +645% return at −1.04% max drawdown with zero costs
is a **trade-mechanics artifact** of the 1% trailing stop + same-bar close exits + immediate re-entry,
not evidence of predictive edge. The repo's own `v2_vs_v3_summary.md` already says as much.

### Concrete suggestions
1. Always net out costs: run `run_execution_aware_replay` for V3-5 too, and fix the reason mapping so
   stop slippage actually applies (map `risk_exit` → `trailing_stop`).
2. Compute accuracy and returns in **one** process, on **one** bar pull and **one** split; pin the
   window explicitly (an `end_date`) instead of copying numbers between runs.
3. Add purge/embargo ≥ the label horizon at split boundaries.
4. Persist per-trade events, equity curve, and the confusion matrix for the exact return window.
5. Report a no-trailing / wider-trailing control so the model's contribution is separable from the
   trailing-stop mechanic, and show cost sensitivity.

## Part 2 — V3-Q3E design (how it addresses the above)

Module: `src/algoding/execution/qwen_embedding_directional_research.py`;
CLI: `python -m algoding.cli qwen-embedding-v3-q3e-run`.

- **Encoder**: frozen `Qwen/Qwen3-Embedding-4B`, 4-bit (nf4) on CUDA, last-token pooled + L2 normalized,
  with an on-disk embedding cache keyed by **SHA256** of the formatted text + metadata (the previous
  scaffold used Python `hash()`, which is salted per process and so was not even stable — fixed).
  Automatic OOM fallback to `batch_size=1, max_length=512`.
- **Labels**: 5-day forward direction (bullish/neutral/bearish) from the existing volatility-scaled
  band; balanced class weights + balanced sampling to counter neutral skew.
- **Variants**: `price_only_baseline`, `q3e_text_only`, `q3e_fused` (embedding + price/risk + DeepSeek
  sentiment when its cache coverage is adequate), `q3e_sector` (per-sector classifiers, pooled fallback).
- **Leakage controls**: chronological split **plus** purge/embargo (drop the last `embargo_days` of
  train and validation); features use only `close[T]` and past bars; news timing unchanged.
- **Trade mapper (parity with V3-5)**: 5% stop, 1% trailing, no-same-day-reentry, both `medium_or_high`
  and `high_only` thresholds, no momentum confirmation. Every variant **and** a freshly-replayed V3-5
  baseline are scored on the **identical** test window (window capped at the V3-5 cache's last day), and
  reported **gross and net** of the repo's standard slippage/fee model.
- **Outputs**: `v3_q3e_run.json`, `v3_q3e_summary.md`, per-symbol accuracy / balanced accuracy / macro F1
  / confusion matrices, gross+net return and max drawdown, the V3-5 comparison table, and
  return/drawdown bar charts (matplotlib).
