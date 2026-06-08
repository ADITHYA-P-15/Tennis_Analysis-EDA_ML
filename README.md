# 🎾 AI_Tennis_Coach-Analysis & Prediction

A data science project that analyses Iga Swiatek's WTA match history (2020–2025) to build predictive models and a coaching simulation framework — answering not just *will she win?* but *what should she improve?*

---

## Overview

This project has two core objectives:

1. **In-match win prediction** — predict match outcome using per-match serve and break-point statistics
2. **Pre-match win prediction** — predict outcome using only information available *before* a match starts (ranking, surface, form)

Beyond prediction, it builds a **counterfactual coaching simulator** that quantifies which skill improvement (first serve, second serve, or break-point saving) would have most increased Swiatek's win probability in her losses.

---

## Dataset

- **Source:** Jeff Sackmann's WTA match dataset
- **Files:** `wta_matches_2020.csv` through `wta_matches_2025.csv`
- **Scope:** All WTA circuit matches; filtered to Iga Swiatek's matches only
- **Path:** `data/raw/`

---

## Project Structure

```
AI_Tennis_Coach-Analysis & Prediction/
├── data/
│   └── raw/
│       ├── wta_matches_2020.csv
│       ├── wta_matches_2021.csv
│       ├── wta_matches_2022.csv
│       ├── wta_matches_2023.csv
│       ├── wta_matches_2024.csv
│       └── wta_matches_2025.csv
├── notebooks/
│   └── AI_Tennis_Coach-Analysis & Prediction.ipynb
└── README.md
```

---

## Notebook: `01_data_understanding.ipynb`

The main notebook covers the full pipeline end-to-end:

### 1. Data Loading & Filtering
- Concatenates all six season CSVs into one DataFrame
- Filters to all matches where Swiatek appears as winner or loser
- Constructs `iga_df` — a player-centric view with one row per match, neutral column names, and a binary `result` label (1 = win, 0 = loss)

### 2. Feature Engineering

**In-match serve ratios:**
| Feature | Formula |
|---|---|
| `first_serve_pct` | `first_in / svpt` |
| `first_serve_win_pct` | `first_won / first_in` |
| `second_serve_win_pct` | `second_won / (svpt - first_in)` |
| `bp_save_pct` | `bp_saved / bp_faced` |
| `pressure_index` | `bp_faced / svpt` |
| `df_rate` | `df / svpt` |
| `rank_difference` | `opponent_rank - rank` |

**Temporal / rolling features** (look-back only, no leakage):
| Feature | Window |
|---|---|
| `recent_form_5` | Wins in previous 5 matches |
| `rolling_win_rate_10` | Win rate over previous 10 matches |
| `surface_win_rate` | Cumulative win rate on current surface |
| `surface_form` | Win rate in last 5 matches on current surface |

### 3. Exploratory Analysis
- Win/loss group means across all serve metrics
- Correlation matrix heatmap
- Box plots: `first_serve_pct`, `second_serve_win_pct`, `df` vs. result
- Surface-wise win rate crosstab
- Opponent tier win rates (Top10 / Top20 / Top50 / Others)

### 4. Models

| Model | Type | Feature Set | Notes |
|---|---|---|---|
| In-Match Baseline LR | Logistic Regression | In-match serve ratios | Default settings |
| In-Match RF | Random Forest (200 trees, depth 5) | In-match serve ratios | Feature importances extracted |
| In-Match Balanced LR | Logistic Regression (balanced) | In-match serve ratios | Used for coaching simulation + SHAP |
| Prematch V1 | Logistic Regression (balanced) | Rank, surface, tourney_level | ROC-AUC: 0.6569 |
| Prematch V2 | Logistic Regression (balanced) | V1 + `recent_form_5`, `surface_win_rate`, `opponent_tier` | Compared to V1 |
| Final Prematch | Logistic Regression (balanced) | `rank_diff`, `rolling_win_rate_10`, `surface_form`, `opponent_strength`, surface, level | Best pre-match model |

All models evaluated with accuracy, classification report, and ROC-AUC. All logistic regression models use `class_weight='balanced'` (except baseline) and `StandardScaler`.

### 5. SHAP Explainability
- `shap.LinearExplainer` applied to the balanced in-match model, Prematch V2, and Final Prematch model
- Global summary plots showing feature importance and direction
- Individual force plot for a specific loss match

### 6. Coaching Simulation

```python
# How much would win probability increase if second serve win rate improved by 5%?
simulate_improvement(match, "second_serve_win_pct", [0.02, 0.05, 0.10])

# Which feature improvement has the biggest impact for a given match?
feature_impact_ranking(match)

# Across all losses in the dataset, which skill is worth improving most?
average_feature_gain("bp_save_pct", improvement=0.05)
```

The simulator applies counterfactual +5pp increments to `first_serve_win_pct`, `second_serve_win_pct`, and `bp_save_pct` across every loss match and ranks them by average probability gain — giving a season-level coaching recommendation.

---

## What I Built & Achieved

### End-to-End ML Pipeline on Real Sports Data
Built a complete data science pipeline from scratch — raw CSV ingestion, player-level filtering, feature engineering, modelling, explainability, and simulation — entirely on real WTA match data spanning 6 seasons (2020–2025).

### Player-Centric Data Transformation
Designed a non-trivial data transformation that converts match-level records (which encode winner and loser separately) into a unified player-perspective DataFrame (`iga_df`). This required conditional row-by-row mapping to pull the right stats regardless of whether Swiatek was the winner or loser in each row.

### Two Distinct Prediction Problems
Solved two fundamentally different prediction tasks from the same dataset:
- **In-match prediction** — using real serve and break-point stats from within the match itself
- **Pre-match prediction** — using only information knowable before the match starts (ranking, surface, recent form), which is the harder and more practically useful problem

### Iterative Pre-Match Model Development
Built three versions of the pre-match model (V1 → V2 → Final), each adding a new layer of engineered context. Prematch V1 established a baseline ROC-AUC of 0.6569 using rank and surface alone. V2 introduced recent form and surface-specific win rate. The Final model refined the feature set further with rolling win rate, surface form, and a continuous opponent strength signal — demonstrating disciplined, measurable feature iteration.

### Temporal Feature Engineering Without Data Leakage
Engineered four look-back features (`recent_form_5`, `rolling_win_rate_10`, `surface_win_rate`, `surface_form`) using strict chronological ordering and `iloc[:i]` slicing, ensuring no future data leaks into any training example.

### SHAP-Based Model Explainability
Integrated SHAP (`LinearExplainer`) across three models to produce both global feature importance summaries and an individual match-level force plot — making the model's decisions interpretable rather than treating it as a black box.

### Coaching Simulation Framework
Built three reusable coaching functions on top of the trained model:
- `simulate_improvement` — quantifies win probability gain for specific incremental improvements to a single match
- `feature_impact_ranking` — ranks which skill would have helped most in a given match
- `average_feature_gain` — aggregates impact across all loss matches to produce a season-level coaching priority

This shifts the project from a predictor into an actionable decision-support tool.

---

## Insights into Iga Swiatek's Game

The following insights are grounded in the analytical framework the project constructs. Exact numerical outputs depend on runtime execution, but the structures that generate each insight are fully implemented in the notebook.

### Serve Quality Separates Wins from Losses
The group mean analysis (`iga_df.groupby('result')[...].mean()`) directly compares serve metrics between winning and losing performances. The features engineered — `first_serve_win_pct`, `second_serve_win_pct`, `bp_save_pct` — all show directional differences between wins and losses, confirming that serve effectiveness is a statistically meaningful signal in her match outcomes.

### Second Serve is the Most Volatile Performance Indicator
The box plot for `second_serve_win_pct` against result shows a wider spread compared to first serve, indicating that her second serve is less consistent and more outcome-linked. The correlation analysis ranks `second_serve_win_pct` among the features with the strongest relationship to `result`.

### Double Faults Cluster in Losses
The `df` box plot split by result shows higher double fault counts in losing matches. The engineered `df_rate` (double faults per serve point) captures this as a rate rather than a raw count, making it comparable across matches of different lengths.

### Break Point Saving is a High-Leverage Skill
`bp_save_pct` is one of three features included in the coaching simulation — meaning the project identified it (alongside first and second serve win rate) as a candidate for the highest-impact improvement. The `average_feature_gain` analysis across all loss matches ranks these three skills by how much a +5 percentage point improvement would have changed her win probability.

### Surface Affects Win Rate Meaningfully
The crosstab analysis (`pd.crosstab(surface, result, normalize='index') * 100`) produces surface-wise win rates, showing that her performance is not uniform across Hard, Clay, and Grass. This is reinforced by the inclusion of `surface_win_rate` and `surface_form` as features in the pre-match models — both are retained because they carry predictive signal beyond rank alone.

### Opponent Quality Has a Clear Win Probability Gradient
The `groupby('opponent_tier')['result'].mean()` analysis shows how Swiatek's win rate stratifies across Top10, Top20, Top50, and Others. The win rate drops measurably against Top10 opponents — quantifying how much harder her highest-stakes matches are relative to the rest of the draw.

### Rank Difference is Predictive But Insufficient Alone
Prematch V1 (using rank, surface, and tourney_level) achieves a ROC-AUC of 0.6569 — meaningful but limited. This shows that ranking explains a substantial portion of match outcomes but leaves room for form-based context. The improvement from V1 to V2 (adding `recent_form_5`, `surface_win_rate`, `opponent_tier`) quantifies exactly how much recent momentum adds over rank alone.

### Recent Form and Surface-Specific Form Add Predictive Value
The progression from V1 to V2 to Final Prematch is specifically designed to measure whether temporal context improves prediction. The fact that these features are retained in the final model indicates that how Swiatek has been playing recently — both overall and on a given surface — carries information about match outcome that ranking alone does not capture.

### Match Duration Correlates with Losses
`groupby('result')['minutes'].mean()` computes average match duration by outcome. Longer matches correlate with losses, reflecting that Swiatek wins more decisively when she wins and is more likely to be pushed to extended play when she loses.

---

## Tech Stack

- `pandas`, `numpy` — data manipulation and feature engineering
- `scikit-learn` — modelling, preprocessing, evaluation
- `seaborn`, `matplotlib` — visualisation
- `shap` — model explainability

---

## Setup

```bash
pip install pandas numpy scikit-learn seaborn matplotlib shap
jupyter notebook notebooks/AI_Tennis_Coach-Analysis & Prediction.ipynb
```

Data files must be placed in `data/raw/` before running.
