# V0 vs V2 on Commodity Contenders

Artifacts:

- [V0 commodity run](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_commodities_daily_run.json)
- [V2 commodity directional run](C:\SVNProjects\Algoding\reports\research\deepseek_directional_commodities_v2\deepseek_directional_run.json)
- [V0 vs V2 comparison JSON](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2.json)

Scope:

- The current commodity DeepSeek branch has `4` active commodity ETF contenders, not `5`: `GLD`, `SLV`, `USO`, `UNG`.
- Strict test period: `2025-08-14 -> 2026-04-02 (160 days)`

Aggregate strict-test comparison:

- `V0` mean total return: `1.90%`
- `V0` mean max drawdown: `-4.84%`
- `V2` mean direction accuracy: `42.35%`
- `V2` mean total return: `47.09%`
- `V2` mean max drawdown: `-1.07%`
- `V2` beat `V0` on return in `4/4` symbols

Per symbol:

- `GLD`
  V0 exact prompt: `2.53%` return, `-4.26%` max drawdown
  V2 finetuned directional: `45.23%` return, `-1.78%` max drawdown, `52.38%` direction accuracy
- `SLV`
  V0 exact prompt: `0.42%` return, `-9.73%` max drawdown
  V2 finetuned directional: `99.52%` return, `-2.07%` max drawdown, `47.92%` direction accuracy
- `USO`
  V0 exact prompt: `5.27%` return, `-3.69%` max drawdown
  V2 finetuned directional: `40.72%` return, `-0.25%` max drawdown, `38.26%` direction accuracy
- `UNG`
  V0 exact prompt: `-0.63%` return, `-1.67%` max drawdown
  V2 finetuned directional: `2.90%` return, `-0.18%` max drawdown, `30.85%` direction accuracy
