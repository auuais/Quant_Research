# DeepSeek Top-5 Strict Exact vs Meta Comparison

Artifact files:

- [Strict comparison JSON](C:\SVNProjects\Algoding\reports\research\deepseek_top5_strict_exact_vs_meta.json)
- [Strict comparison PDF](C:\SVNProjects\Algoding\output\pdf\deepseek_top5_strict_exact_vs_meta_dark.pdf)

This comparison avoids metric mixing:

- `Previous window average` is the earlier broad DeepSeek leaderboard context from the `1y/2y/3y` daily study.
- `Strict exact prompt variant` reruns the exact earlier winning DeepSeek variant on the later unseen test slice only.
- `Strict meta model` is the fine-tuned DeepSeek news+price meta-model on that same unseen test slice.

## Exact earlier winners used

- `AVGO / avgo_momentum_llm_entry_filter`, `entry_sentiment_min = 0.347`
- `NVDA / nvda_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`
- `TSLA / tsla_momentum_llm_exit_filter`, `exit_sentiment_max = 0.812801`
- `GOOGL / googl_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`
- `XOM / xom_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`

## Aggregate

- Earlier multi-window average mean return: `44.60%`
- Strict exact prompt mean return: `10.43%`
- Strict meta-model mean return: `1.06%`

- Earlier multi-window average mean max drawdown: `-10.91%`
- Strict exact prompt mean max drawdown: `-3.11%`
- Strict meta-model mean max drawdown: `-0.03%`

The strict exact prompt variant beat the strict meta-model on return in `5/5` symbols.

## Per symbol

- `AVGO`
  - previous average: `62.16%`, drawdown `-5.59%`
  - strict exact prompt: `5.39%`, drawdown `-3.36%`
  - strict meta: `0.87%`, drawdown `-0.02%`

- `NVDA`
  - previous average: `58.71%`, drawdown `-15.98%`
  - strict exact prompt: `4.83%`, drawdown `-3.92%`
  - strict meta: `4.41%`, drawdown `-0.11%`

- `TSLA`
  - previous average: `51.56%`, drawdown `-21.70%`
  - strict exact prompt: `8.80%`, drawdown `-4.61%`
  - strict meta: `0.00%`, drawdown `0.00%`

- `GOOGL`
  - previous average: `33.57%`, drawdown `-7.84%`
  - strict exact prompt: `23.18%`, drawdown `-1.86%`
  - strict meta: `0.00%`, drawdown `0.00%`

- `XOM`
  - previous average: `17.00%`, drawdown `-3.41%`
  - strict exact prompt: `9.95%`, drawdown `-1.81%`
  - strict meta: `0.00%`, drawdown `0.00%`

## Conclusion

The earlier DeepSeek stock winners remain valid as the headline results for the broad daily multi-window study.

But when rerun on a strict later unseen test slice, returns compress substantially. Even so, the exact earlier DeepSeek prompt variants still outperform the current fine-tuned DeepSeek news+price meta-model on return for all five symbols.

The current fine-tuned branch is more defensive, but still too selective and not yet competitive as the lead stock-news trading branch.
