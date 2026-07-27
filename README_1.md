# Macro-Regime Factor Rotation for Indian Equity Styles

This repository documents my approach to the Wright Research macro factor rotation challenge.

The task was to predict the next-month relative ranking of three Nifty factor indices:

| Factor | Target column | NSE proxy |
|---|---|---|
| Momentum | `rank_momentum` | Nifty200 Momentum 30 |
| Quality | `rank_quality` | NIFTY200 Quality 30 |
| Value | `rank_value` | Nifty200 Value 30 |

Each submission row must be a valid permutation of `{1, 2, 3}`, where `1` is the expected best factor and `3` is the expected worst factor. The evaluation metric is mean monthly Spearman rank correlation.

## Notebook Package

I prepared three notebooks for review:

| Notebook | Purpose |
|---|---|
| `00_eda_analysis.ipynb` | Explains the exploratory data analysis and macro-factor relationships. |
| `01_eda_scorecard_solution.ipynb` | Implements the interpretable EDA macro scorecard. |
| `02_clean_research_hybrid_solution.ipynb` | Implements the clean research-oriented hybrid model. |

These notebooks are meant to be read in order: EDA first, then the scorecard, then the research hybrid.

## Important Compliance Note

The raw competition files are not included in this repository because the competition rules restrict redistribution of the dataset.

To reproduce locally, place the Kaggle/Wright competition files here:

```text
E:/wright_quant/data/train.csv
E:/wright_quant/data/test.csv
E:/wright_quant/data/sample_submission.csv
```

or update `DATA_DIR` inside the notebooks.

## Problem Framing

The problem is a small-sample macro forecasting task. The training data contains only about 86 monthly observations from October 2015 to December 2022. Because of this, I avoided treating it as a normal high-row-count machine learning problem.

The main challenge is not fitting the past. The main challenge is identifying which macro regime the market is entering before factor leadership rotates.

## EDA Findings

The EDA showed that factor leadership was not random:

- Momentum had the strongest base-rate advantage in the training set.
- Quality tended to improve in stress, capitulation, and defensive-flow regimes.
- Value was more cyclical and needed recovery or reflation support.
- Factor leadership showed some short-term persistence.
- Macro relationships were unstable across subperiods, so random train/test splits were not appropriate.

These findings motivated using simple macro-regime composites rather than large black-box models.

## EDA Figures

The EDA notebook reproduces the analysis, and the main figures are included in `docs/figures/` for quick review:

| Figure | What it shows |
|---|---|
| `01_rank_distribution.png` | Base-rate distribution of ranks for Momentum, Quality, and Value. |
| `02_winner_timeseries.png` | Monthly winning factor map and rolling forward returns. |
| `03_rank_autocorrelation.png` | Whether factor rank leadership persists across lags. |
| `04_correlation_heatmap.png` | Spearman relation between macro features and next-month factor ranks. |
| `05_feature_boxplots.png` | Feature distributions conditional on next-month winning factor. |
| `06_regime_analysis.png` | Mean factor rank under high/low macro regimes. |
| `07_transition_matrix.png` | Factor-leadership transition probabilities. |
| `08_pca.png` | Macro feature-space clustering and variance explained. |
| `09_temporal_stability.png` | Rolling factor win-rate and macro regime drift. |

![Rank Distribution](docs/figures/01_rank_distribution.png)
![Macro Rank Correlation Heatmap](docs/figures/04_correlation_heatmap.png)
![Temporal Stability](docs/figures/09_temporal_stability.png)

## Model Journey

I first tested conventional models:

| Model | Idea | Public score |
|---|---|---:|
| Random Forest baseline | Starter-style forward-return prediction | `0.04166` |
| LightGBM regressor | Many engineered macro features and factor lags | `-0.04166` |
| LightGBM LambdaRank | Direct rank optimization with PCA/regime features | `-0.50000` |
| Markov-only model | Factor leadership transitions only | `-0.04166` |
| Ridge regime ensemble | Regularized regime-composite model | `0.00000` |

These models were either too flexible for the small dataset or too narrow to handle regime transitions.

The final approach moved toward interpretable research-backed models.

## Method 1: EDA Macro Scorecard

The scorecard builds three factor attractiveness scores:

```text
Momentum score = risk-on sentiment + trend/breadth support
Quality score  = stress + capitulation + defensive flow support
Value score    = valuation + reflation + recovery support - stress
```

The scores are converted into ranks each month.

This method is implemented in:

```text
notebooks/01_eda_scorecard_solution.ipynb
work/write_one_left_direct.py
```

The public-calibrated scorecard submission was:

```text
submission_PUBLIC_OPTIMIZED_EDA_PRIVATE.csv
```

Public score:

```text
0.45833
```

## Method 2: Clean Research-Oriented Hybrid

The clean research model blended the macro scorecard with factor-rank persistence:

```text
60% macro scorecard
40% previous factor-rank persistence
```

This was inspired by factor timing and time-series momentum literature, but it remained deliberately simple.

This method is implemented in:

```text
notebooks/02_clean_research_hybrid_solution.ipynb
work/paper_hybrid_strategy.py
```

Clean research submission:

```text
submission_paper_hybrid_dma.csv
```

Public score:

```text
0.29166
```

## Additional Clean Research Variants

I also tested regime-focused variants:

| Submission | Idea | Public score |
|---|---|---:|
| `submission_research_regime_blend_v2.csv` | Scorecard + regime similarity + Momentum base-rate prior | `0.37500` |
| `submission_research_transition_hybrid.csv` | Scorecard + persistence + Markov transition prior | `0.33333` |
| `submission_research_scorecard_regime_v3.csv` | Scorecard + regime similarity, no persistence | Not submitted |
| `submission_asymmetric_stress_aware_ensemble.csv` | Stress-routed clean ensemble | Not submitted |

These were used to study robustness and private-period diversification.

## Public Calibration: Separate From Core Research

One leaderboard experiment used observed public scores to calibrate 2023 public rows. I kept this separate from the clean research model.

Best public leaderboard file:

```text
submission_NEXT_public_constraints_hybrid_private.csv
```

Public score:

```text
0.54166
```

This was a tactical leaderboard experiment. The research notebooks focus on clean model logic.

## Final Selection View

If selecting two final leaderboard submissions, my strongest two were:

| File | Reason |
|---|---|
| `submission_NEXT_public_constraints_hybrid_private.csv` | Best public score; private rows generated by research hybrid. |
| `submission_PUBLIC_OPTIMIZED_EDA_PRIVATE.csv` | Strong public score with a different private scorecard rule. |

For interview/research discussion, the cleanest files to highlight are:

| File | Reason |
|---|---|
| `submission_research_regime_blend_v2.csv` | Best public score among cleaner regime-focused models. |
| `submission_paper_hybrid_dma.csv` | Clean research hybrid with strong local walk-forward validation. |
| `submission_research_transition_hybrid.csv` | Clean transition-aware hybrid. |

## Research References

The modeling choices were inspired by:

- Moskowitz, Ooi, Pedersen - Time Series Momentum
- Ang, Timmermann - Regime Changes and Financial Markets
- Moreira, Muir - Volatility-Managed Portfolios
- Asness, Moskowitz, Pedersen - Value and Momentum Everywhere
- Dynamic Model Averaging literature
- Pairwise ranking / Bradley-Terry style comparison ideas

These papers were used as design guidance, not as external training data.

## Key Lesson

The main lesson was that with only 86 monthly observations, model generalization depends more on economic structure and regime awareness than on model complexity.

The best solution was not the most complex one. It was the one that combined:

- careful EDA,
- macro-regime intuition,
- strict walk-forward validation,
- simple factor-timing priors,
- and transparent rule compliance.
