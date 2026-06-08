# 🎾 AI_Tennis_Coach-Analysis & Prediction

A data science and machine learning project that analyses Iga Swiatek's WTA match history (2020–2025) to build predictive models and a coaching simulation framework — answering not just *will she win?* but *what should she improve?*

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
