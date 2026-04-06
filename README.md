# Spotify Analysis Capstone Project
### What Music Is Worth Signing? An Exploratory Analysis Backed by Regression Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=flat-square&logo=pandas)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Hypothesis Testing Summary](#hypothesis-testing-summary)
- [Machine Learning Results](#machine-learning-results)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Column Reference](#column-reference)

---

## Project Overview

This capstone project presents a data-driven analytical framework designed to support a mid-sized record label in making informed quarterly signing decisions. Using a 114,000-track Spotify dataset spanning 114 genres, the project moves through four structured stages: data cleaning, exploratory data analysis (EDA), data preprocessing, and machine learning.

The central question driving every stage of this analysis is: **what kind of music should the record label bet on?** The answer is built from statistical hypothesis testing, multivariate genre and artist-level investigation, and two regression models (Random Forest and XGBoost) whose feature importance scores independently validated the findings from the EDA.

---

## Team Members

| Name | Email | GitHub | Role |
|---|---|---|---|
| **PRINCESS CHIAMAKA EMENARI** | princessemenari2@gmail.com | [Github](https://github.com/Princess-ma) | ***Team Lead*** |
| **Samuel Adewumi** | adewunmisamuel584@gmail.com | [Github](https://github.com/adewunmisamuel584-dev) | Team Member |

---

## Problem Statement

A mid-sized record label is planning a new quarterly signing cycle and needs data-backed guidance on what kind of music to invest in. The major aim of this project is to analyse the Spotify dataset and provide meaningful insights that could direct the label's decision-making on new signings.

---

## Dataset

**Source:** [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

**Size:** 114,000 rows, 21 columns

**Structure:**

| Property | Value |
|---|---|
| Total rows | 114,000 |
| Total columns | 21 |
| Float columns | 9 |
| Integer columns | 6 |
| Categorical / text columns | 5 |
| Boolean columns | 1 |
| Columns with missing values | 3 (`artists`, `album_name`, `track_name` - 1 row each) |
| Unique genres | 114 (1,000 tracks per genre) |
| Unique artists | 31,437 |
| Unique track IDs | 89,741 |

The same track can appear under multiple genres, which is an intentional design of the dataset and not a data error. All 114,000 rows were retained after confirming that zero full-row duplicates exist.

---

## Project Objectives

1. Clean and prepare the dataset for exploratory analysis.
2. Identify and visualise patterns and trends that popular music follows.
3. Preprocess the data by selecting meaningful features for the regression models.
4. Build a baseline **Random Forest Regressor** and an advanced **XGBoost Regressor** to predict popularity and identify which features drive it.
5. Translate all findings into actionable recommendations for the record label's signing strategy.

---

## Methodology

The project was executed in four sequential stages.

**Stage 1: Data Cleaning and Preparation**
The dataset was loaded, inspected for shape, data types, missing values and duplicates. Three columns (`artists`, `album_name`, `track_name`) each had exactly one missing row at 0.00087% of the dataset. These rows were left intact during EDA and dropped during preprocessing. Two engineered features were created on a working copy of the dataset (`spp`): `duration_min` (milliseconds converted to minutes) and `duration_range` (minute intervals bucketed into labelled categories). No duplicate rows were found across all 114,000 entries.

**Stage 2: Exploratory Data Analysis**
The EDA operated across three sub-stages. The data distribution stage examined all 21 columns through value counts, unique value analysis, percentage distributions, histograms, boxplots and bar charts. The hypothesis testing stage ran 14 targeted statistical tests (ANOVA, Pearson correlation, and independent samples t-tests) to quantify the relationship between each feature and the target variable, popularity. The exploratory analysis stage investigated genre-level hit production versus average popularity consistency, audio feature profiles of top genres, artist-level dominance, and the overlap between popular artists and the pop genre.

**Stage 3: Data Preprocessing**
Fifteen columns were dropped from the modelling dataset, including all identity columns, the redundant millisecond duration column, and all ten category columns created during EDA. The three remaining categorical columns were encoded: `explicit` was cast from boolean to integer, and `artists` and `track_genre` were label-encoded into `artists_encoded` and `genre_encoded`. Seven features identified as commercially irrelevant through hypothesis testing (`energy`, `liveness`, `mode`, `key`, `time_signature`, `tempo`, and the high-skew `instrumentalness` and `speechiness` prior to log transformation) were handled accordingly. Log transformation using `np.log1p()` was applied to `speechiness` and `instrumentalness` before all ten final features were standardised with `StandardScaler`. The dataset was split 80/20 into training and test sets.

**Stage 4: Machine Learning**
A tuned Random Forest Regressor served as the baseline model. A tuned XGBoost Regressor served as the advanced model. Both were tuned using `RandomizedSearchCV`. Feature importance scores from both architectures were compared against the EDA findings.

---

## Key Findings

**Genre is the single most powerful predictor of popularity.** The ANOVA F-statistic of 343.462 dwarfed every other result in the analysis, confirming that the genre a track belongs to is the primary commercial differentiator on Spotify.

**Instrumentalness is the strongest individual audio feature finding.** A Pearson correlation of -0.095 confirmed that vocal driven, professionally produced music consistently outperforms instrumental and acoustic content on the platform.

**Pop dominates in hit volume but not in consistency.** Pop produced 131 tracks above the ***80 popularity threshold**, the highest of any genre. However, only 13.1% of pop tracks cross that threshold. Genres like pop-film (avg. popularity 59.28) and k-pop (56.90) outperform pop on average popularity despite fewer breakout moments.

**Latin and reggaeton represent the highest-return signing ecosystem.** Bad Bunny alone produced 42 popular tracks, nearly double the second placed Red Hot Chili Peppers at 23. He appears six times in the top 20 artists by average popularity as a solo act and in collaborations.

**Popular artists and pop artists are almost entirely different groups.** Only 5.5% of artists in the top 25% by average popularity have any tracks classified under the pop genre. The platform's most commercially successful performers are spread across genres far more broadly than the genre charts suggest.

**Tracks in the 2–5 minute duration window score 3.6 popularity points higher** than tracks outside that range on average.

**Explicit tracks score 3.5 popularity points higher** than non-explicit tracks on average.

**Energy, liveness and tempo's BPM range showed no statistically significant relationship with popularity** and should not factor into production or signing direction decisions.

---

## Hypothesis Testing Summary

| # | Hypothesis | Test | Key Result | P-Value | Decision | Practical Significance |
|---|---|---|---|---|---|---|
| H1 | Genre vs Popularity | ANOVA | F = 343.462 | 0.00000 | Reject H0 | Very High |
| H2 | Danceability vs Popularity | Pearson | r = 0.035 | 0.00000 | Reject H0 | Very Low (sample-size driven) |
| H3 | Explicitness vs Popularity | T-test | Explicit 36.45 vs 32.94 | 0.00000 | Reject H0 | Moderate |
| H4 | Energy vs Popularity | Pearson | r = 0.001 | 0.72140 | Fail to Reject H0 | None |
| H5 | Duration vs Popularity | T-test | 2–5 min: 33.95 vs 30.35 | 0.00000 | Reject H0 | Moderate |
| H6 | Valence vs Popularity | Pearson | r = -0.041 | 0.00000 | Reject H0 | Very Low |
| H7 | Loudness vs Popularity | Pearson | r = 0.050 | 0.00000 | Reject H0 | Low |
| H8 | Mode vs Popularity | T-test | Minor 33.65 vs Major 33.00 | 0.00000 | Reject H0 | Very Low |
| H9 | Artist Track Volume vs Popularity | Pearson | r = -0.072 | 0.00000 | Reject H0 | Low-Moderate |
| H10 | Speechiness vs Popularity | Pearson | r = -0.045 | 0.00000 | Reject H0 | Very Low |
| H11 | Acousticness vs Popularity | Pearson | r = -0.025 | 0.00000 | Reject H0 | Very Low |
| H12 | Instrumentalness vs Popularity | Pearson | r = -0.095 | 0.00000 | Reject H0 | Moderate (strongest audio feature) |
| H13 | Liveness vs Popularity | Pearson | r = -0.005 | 0.06893 | Fail to Reject H0 | None |
| H14 | Tempo (continuous) | Pearson | r = 0.013 | 0.00001 | Reject H0 | Very Low (negligible) |
| H14 | Tempo (90–120 BPM range) | T-test | Delta mean = 0.177 | 0.21169 | Fail to Reject H0 | None |

---

## Machine Learning Results

### Model Comparison

| Model | Train R² | Test R² | Train MAE | Test MAE | Test % Error | Train-Test Gap |
|---|---|---|---|---|---|---|
| Random Forest (Untuned) | 0.8444 | 0.5179 | 6.066 | 11.284 | 33.95% | 0.327 |
| Random Forest (Tuned) | 0.9468 | 0.5515 | 2.451 | 10.256 | 30.86% | 0.395 |
| XGBoost (Untuned) | 0.5365 | 0.4459 | 11.162 | 12.231 | 36.80% | 0.091 |
| XGBoost (Tuned) | 0.8042 | 0.5749 | 6.827 | 10.337 | 31.10% | 0.229 |

The **tuned XGBoost** is the recommended model. It achieved the highest test R² of 0.5749 and the smallest training-to-test gap among all tuned models at 0.229 points, confirming stronger generalisation to unseen data.

### Feature Importance (Tuned Models)

Both models independently ranked genre and artist encoding as the first and second most important features, directly validating the EDA findings. The results below are from the tuned XGBoost.

| Feature | Importance |
|---|---|
| genre_encoded | 0.2662 |
| explicit | 0.1278 |
| artists_encoded | 0.1008 |
| acousticness | 0.0814 |
| duration_min | 0.0756 |
| instrumentalness | 0.0732 |
| danceability | 0.0711 |
| valence | 0.0697 |
| loudness | 0.0688 |
| speechiness | 0.0654 |

The 42.51% of unexplained variance reflects the influence of external factors absent from the dataset: marketing activity, playlist placement, social media dynamics, and algorithmic promotion. This is a ceiling of what audio features and genre identity can explain on their own, not a failure of the models.

### Tuning Configuration

**Best Random Forest Parameters:** `n_estimators=300`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `max_depth=30`, `bootstrap=False`

**Best XGBoost Parameters:** `n_estimators=700`, `max_depth=7`, `learning_rate=0.15`, `subsample=0.8`, `colsample_bytree=0.6`, `gamma=0.1`, `min_child_weight=7`, `reg_lambda=0.5`, `reg_alpha=0.5`

---

## Recommendations

**Priority 1: Focus on Latin and reggaeton.** Every layer of the analysis converges on this ecosystem as the most commercially productive. Priority should be given to artists with collaborative histories, as collaborative Latin acts generate multiple independently strong catalogue entries.

**Priority 2: Allocate budget to consistency driven genres.** Genres like k-pop (avg. popularity 56.90), pop-film (59.28), chill (53.65) and sad (52.38) maintain popularity floors that exceed pop itself. These genres offer a lower-risk path for building catalogue longevity.

**Priority 3: Evaluate artists by average popularity across at least three tracks.** Single-release viral outliers inflate perceived commercial potential and do not represent the sustainable performance level a label needs to plan around.

**Priority 4: Prioritise vocal-driven, produced artists in the 2–5 minute duration window.** The hypothesis testing, audio feature profiles of top genres and machine learning feature importance scores all converge on the same production profile.

**Priority 5: Cross-reference hit count and average popularity simultaneously.** The dance genre produced 107 tracks above 80 popularity but an average popularity of only 22.69, exposing extreme bimodality. High hit count does not indicate broad catalogue reliability.

**Deprioritise the following genres entirely:** romance, iranian, tango, grindcore, detroit-techno, salsa, chicago-house, and similar genres where 98–100% of tracks fall below a popularity score of 40.

---

## Limitations

- The Spotify popularity score is a continuously updating metric tied to the moment of data collection. Rankings may shift as streaming behaviour evolves.
- The perfectly balanced 1,000-tracks-per-genre design does not reflect the true proportional distribution of music on Spotify. Findings about smaller genres should be interpreted with caution.
- Label encoding of the `artists` column imposes an implicit ordinal assumption on a categorical variable. Tree-based models are less susceptible to this limitation, but artist importance scores should not be interpreted as a direct ranking of artist quality.
- The 14.05% of tracks with a popularity score of zero may influence model predictions toward optimism, reducing accuracy at the low end of the popularity distribution.
- Key factors influencing platform popularity such as marketing spend, playlist placement, social media following, release timing and label affiliation are absent from the dataset. The model should be used as a directional tool, not a definitive verdict.

---

## Tech Stack

```
Python 3
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
xgboost
Google Colab (execution environment)
```

## Column Reference

### Identity Columns

| Column | Type | Description |
|---|---|---|
| `Unnamed: 0` | Integer | Sequential row number. Dropped before modelling. |
| `track_id` | String | Spotify unique track identifier. 21% are duplicated by design (same track, multiple genres). |
| `artists` | String | Artist name(s). 31,437 unique artists. |
| `album_name` | String | Album title. 46,589 unique albums. |
| `track_name` | String | Track title. 73,608 unique titles. |
| `track_genre` | String | Genre. 114 unique genres, each with exactly 1,000 tracks. |

### Target Variable

| Column | Type | Description |
|---|---|---|
| `popularity` | Integer (0–100) | Spotify popularity score. Tracks scoring 80 and above are classified as popular in this project. |

### Audio Feature Columns

| Column | Type | Range | Description |
|---|---|---|---|
| `duration_ms` | Integer | — | Track length in milliseconds. Converted to `duration_min` for analysis. |
| `explicit` | Boolean | True / False | Whether the track contains explicit content. |
| `danceability` | Float | 0.0–1.0 | Suitability for dancing. |
| `energy` | Float | 0.0–1.0 | Intensity and activity level. |
| `key` | Integer | 0–11 | Musical key (Pitch Class notation: 0=C, 1=C#, ... 11=B). |
| `loudness` | Float | -60 to 0 dB | Overall loudness. Values closer to 0 are louder. |
| `mode` | Integer | 0 or 1 | 1 = Major (bright), 0 = Minor (dark). |
| `speechiness` | Float | 0.0–1.0 | Presence of spoken words. Above 0.66 = spoken word; below 0.33 = mostly music. |
| `acousticness` | Float | 0.0–1.0 | Likelihood of acoustic instrumentation. |
| `instrumentalness` | Float | 0.0–1.0 | Likelihood of no vocal content. Above 0.5 is likely instrumental. |
| `liveness` | Float | 0.0–1.0 | Presence of a live audience. |
| `valence` | Float | 0.0–1.0 | Musical mood. High = happy/euphoric; Low = sad/angry. |
| `tempo` | Float | BPM | Track speed in beats per minute. |
| `time_signature` | Integer | 0–5 | Beats per bar. |

---

> **Dataset Confidence Check:** A custom heuristic applied to the dataset returned a **70.0% confidence** that it is a real-world dataset, validating that the conclusions and recommendations drawn from this project can be applied in real-world signing scenarios.

The tuned XGBoost model achieved a **test R² of 0.5749**, explaining approximately 57.49% of popularity variance on unseen tracks using ten features derived from audio characteristics, genre and artist identity. 

---

## Acknowledgements

- **Michael Oluwole** - Project Supervisor and Tutor
- **Kaggle** - for the Spotify Dataset
- **Scikit-learn team** - for developing machine learning libraries
- **GitHub** - for the deployment platform of our Spotify Project

---

## References
Seufitelli, D. B., Oliveira, G. P., Silva, M. O., Scofield, C., & Moro, M. M. (2023). Hit song science: A comprehensive survey and research directions. Journal of New Music Research, 52(1), 41–72. https://doi.org/10.1080/09298215.2023.2282999

Sebastian, N., Mayer, F., Reisz, N., Aulbach, R., Anke, L. T., & Thurner, S. (2024). Beyond beats: A recipe to song popularity? A machine learning approach. arXiv. https://arxiv.org/abs/2403.12079

---

**Notebook Link**: Click on [Spotify Analysis](https://colab.research.google.com/drive/1NE0g5jFSaH1oyJvBSsFz_9s9WuEDt0j8?usp=sharing) to sccess the notebook containing the analysis and findings.

---

**License:** Apache License 2.0
**Copyright:** © 2026 Princess Emenari & Samuel Adewunmi Spotify Analysis Project

<div align="center">

*Built with precision. Validated with evidence.*

</div>
