# **Predicting Medium Article Engagement using Leakage-Aware NLP and Gradient Boosting**

## **Short project summary**

In this project I predict Medium article engagement using article text, metadata, author/publication information, and publication timing. The modeling target is **`log1p(claps)`**, not raw clap counts directly.

The final selected model is a **validation-calibrated Ridge/LightGBM blend**. In the executed full-data notebook run, the blend achieved:

| Model | Validation MAE | Held-out test MAE | Test prediction mean |
|---|---:|---:|---:|
| Ridge/LightGBM blend | **1.09060** | **1.02867** | **2.12045** |
| LightGBM | 1.12090 | 1.04809 | 2.17611 |
| Selected Ridge | 1.15361 | 1.10715 | 2.17590 |

All metrics are MAE on the **`log1p(claps)`** scale. Lower is better.

This project includes data cleaning, target exploration, feature engineering, chronological validation, leakage-aware model training, model comparison, blending, final held-out test evaluation, error analysis, feature importance, ablation testing, and saved prediction outputs.


## **Problem statement**

Medium clap counts are highly skewed. Many articles receive no claps or very few claps, while a small number of viral articles receive thousands of claps. This project asks:

> Can we predict the engagement level of future Medium articles from article text, metadata, author/publication context, and publication timing?

Because raw clap counts are long-tailed, I predict:

```text
log_claps = log(1 + claps)
```

This compresses extreme viral values while preserving the ranking and relative scale of engagement.

The goal is not to build a production-ready Medium ranking system. The goal is to demonstrate a careful, reproducible machine learning workflow for a realistic long-tailed regression problem.


## **Dataset**

The project uses the Medium article dataset:

```text
Alaamer/medium-articles-posts-with-content
```

The dataset contains Medium articles with fields such as:

- title and subtitle
- article text
- tags and metadata
- author and publication information
- publication timestamps
- reading time, word count, image count, link count, and other article-level metadata
- clap counts and related engagement fields

The raw dataset loaded in the notebook contains:

```text
444,593 rows
56 columns
```

For this project, I keep a strict supervised subset of articles that are explicitly labeled as English and have valid clap targets.

Final modeling subset:

```text
66,088 English articles
0 duplicate post IDs
0 missing or invalid clap targets
0 missing publication timestamps
```

The dataset covers articles from:

```text
2010-10-27 to 2018-09-30
```

Most usable English articles are from 2017 and 2018.


## **Target variable**

The raw target is:

```text
claps
```

I use `totalClapCount` as the preferred source column when available.

The main modeling target is:

```text
log_claps = log(1 + claps)
```

The project evaluates models using MAE on the `log1p(claps)` scale. Predictions can be transformed back to approximate clap counts with:

```text
predicted_claps = exp(pred_log_claps) - 1
```

However, the reported model metrics are not raw-clap MAE. They are log-scale MAE.

The target is strongly long-tailed:

| Target characteristic | Value |
|---|---:|
| Zero-clap articles | 31.33% |
| Articles with <= 1 clap | 39.94% |
| Articles with >= 1,000 claps | 2.62% |

I also create a clipped target for experiments:

```text
log_claps_clipped
```

The clipped target caps extreme `log_claps` values at the training-split 99.5th percentile. In the final executed run, the training-only clipping threshold was:

```text
8.59287
```

The final evaluation is still performed against the original `log_claps` target.


## **Project workflow**

The notebook follows this workflow:

1. Load the raw Medium dataset.
2. Create a local Parquet cache for reproducibility and faster reruns.
3. Filter to a clean English-language supervised subset.
4. Parse and validate the clap target.
5. Create canonical text, metadata, author, publication, and timestamp columns.
6. Run dataset diagnostics.
7. Explore the raw and log-transformed target distributions.
8. Engineer row-level features.
9. Build a chronological train/validation/test split.
10. Construct leakage-free sparse feature matrices.
11. Train and tune a Ridge regression baseline.
12. Test validation-based mean calibration for Ridge.
13. Evaluate the selected Ridge benchmark on the held-out test set.
14. Train and evaluate a clipped-target Ridge experiment.
15. Train a LightGBM model using compact engineered, target-encoded, and SVD-compressed sparse features.
16. Select a Ridge/LightGBM blend using validation MAE.
17. Train the final LightGBM workflow on train + validation.
18. Evaluate the final blend on the held-out chronological test set.
19. Run error analysis.
20. Inspect LightGBM feature importance.
21. Run a target-encoding ablation.
22. Save final model comparison and held-out test prediction outputs.
23. Summarize limitations and future improvements.


## **Chronological train/validation/test split**

The notebook uses a chronological split instead of a random split. This better simulates the real task of training on earlier articles and predicting engagement for later articles.

| Split | Rows | Fraction | Date range |
|---|---:|---:|---|
| Train | 46,261 | 70% | 2010-10-27 to 2018-06-21 |
| Validation | 9,913 | 15% | 2018-06-21 to 2018-08-13 |
| Test | 9,914 | 15% | 2018-08-13 to 2018-09-30 |

The split also reveals temporal drift:

| Split | Mean `log_claps` | Median `log_claps` |
|---|---:|---:|
| Train | 2.45630 | 2.07944 |
| Validation | 2.30318 | 1.79176 |
| Test | 2.08826 | 1.38629 |

The later test period has lower engagement on average, which makes the benchmark more realistic and more challenging.


## **Feature engineering**

The notebook creates several groups of features.

### **Text features**

Sparse TF-IDF features are built from:

- `body_text`
- `title_text`
- `metadata_text`

The sparse validation feature space contains:

| Feature block | Number of features |
|---|---:|
| Body TF-IDF | 150,000 |
| Title TF-IDF | 60,000 |
| Metadata TF-IDF | 15,815 |
| Author categorical features | 9,906 |
| Publication categorical features | 2,885 |
| Engineered numeric features | 65 |
| Total sparse features | 238,671 |

### **Numeric and metadata features**

The notebook also engineers numeric features from article and author/publication metadata, including:

- text length features
- word count and reading time features
- image, link, tag, code-block, and audio-duration features
- missing-value indicators
- `log1p` transformations of skewed numeric fields
- ratio features
- publication-time features
- cyclical time encodings
- author and publication frequency features

### **LightGBM compact features**

LightGBM is not trained directly on the full sparse matrix. Instead, it uses a compact feature matrix with:

- engineered numeric features
- author/publication frequency features
- smoothed author/publication target encodings
- 150 SVD-compressed components derived from the sparse Ridge feature matrix

The LightGBM validation feature matrix has:

```text
223 features
```


## **Modeling approach**

### **Ridge regression baseline**

Ridge regression is used as the first strong baseline because it works well with high-dimensional sparse TF-IDF features.

Best Ridge validation setup:

```text
alpha = 10.0
prediction clipping enabled
validation MAE = 1.15881
```

A validation-based mean shift slightly improved Ridge:

```text
Ridge validation MAE before shift = 1.15881
Ridge validation MAE after shift  = 1.15528
```

### **Clipped-target Ridge experiment**

The notebook also tests Ridge trained on `log_claps_clipped`, where the most extreme target values are capped using the training-split 99.5th percentile threshold.

Best clipped-target Ridge result:

```text
alpha = 10.0
post-processing = lower_upper_clip
validation MAE = 1.15361
held-out test MAE = 1.10715
```

This became the selected Ridge baseline.

### **LightGBM model**

LightGBM is trained on compact engineered, target-encoded, and SVD-compressed sparse features.

Best LightGBM validation setup:

```text
config = l1_medium
objective = regression_l1
best iteration = 242
validation MAE = 1.12090
held-out test MAE = 1.04809
```

LightGBM outperformed the selected Ridge baseline on validation and test.

### **Ridge/LightGBM blending**

The final model blends Ridge and LightGBM validation predictions:

```text
blend = w * Ridge + (1 - w) * LightGBM
```

The blend weight is selected using validation MAE.

Final selected blend:

```text
Ridge weight = 0.32
LightGBM weight = 0.68
validation-selected mean shift = -0.06047
```

The final blend achieved the best validation and held-out test performance.


## **Final results**

Final executed full-data results:

| Model | Validation MAE | Held-out test MAE | Test prediction mean |
|---|---:|---:|---:|
| Ridge/LightGBM blend | **1.09060** | **1.02867** | **2.12045** |
| LightGBM | 1.12090 | 1.04809 | 2.17611 |
| Selected Ridge | 1.15361 | 1.10715 | 2.17590 |
| LightGBM without target encoding | 1.19271 | N/A | N/A |

The true test target mean was:

```text
2.08826
```

The final blend prediction mean was:

```text
2.12045
```

The final selected model is the **shifted Ridge/LightGBM blend**.

The result should be interpreted as a strong leakage-aware benchmark on this dataset, not as a production-ready popularity predictor.


## **Error analysis and model limitations**

The final blend performed well overall but still shows the typical weaknesses of popularity prediction.

Final test error summary:

| Metric | Value |
|---|---:|
| Test MAE | 1.02867 |
| Mean residual | 0.03220 |
| Median residual | 0.06424 |
| Median absolute error | 0.71680 |

The error analysis showed a clear pattern:

- The model tends to **overpredict zero-clap and very-low-clap articles**.
- The model tends to **underpredict medium-to-high-clap articles**.
- The model strongly underpredicts **rare viral articles**.
- Predictions are pulled toward the center of the distribution.

This is expected because Medium popularity is highly long-tailed. A small number of viral articles can receive far more claps than typical articles, and those extreme outcomes are difficult to predict from text and metadata alone.

The target-bin analysis showed that the highest-clap bin had the largest error:

| True target range | Rows | Mean residual | MAE |
|---|---:|---:|---:|
| `0` | 3,905 | +0.73038 | 0.73038 |
| `(0, 1]` | 687 | +0.62479 | 0.92121 |
| `(1, 2]` | 891 | +0.52223 | 1.18294 |
| `(2, 4]` | 1,942 | -0.33706 | 1.24312 |
| `(4, 6]` | 2,097 | -0.98082 | 1.20366 |
| `(6, 8]` | 370 | -1.75561 | 1.76896 |
| `> 8` | 22 | -3.02344 | 3.02344 |

The model is strongest for the central mass of articles and weakest at the extremes.

### **Important limitations**

This project is not presented as production-ready. Important limitations include:

1. **Long-tailed target behavior**  
   Rare viral articles are difficult to predict and are often underpredicted.

2. **Zero-clap articles**  
   The model sometimes overpredicts articles that receive zero claps, especially when their metadata resembles more popular articles.

3. **Target encoding dependence**  
   Target encoding is powerful but depends heavily on historical author/publication performance. Performance may drop for brand-new authors or publications.

4. **Potential scrape-time metadata**  
   Some author metadata, such as follower counts, may reflect scrape-time values rather than publication-time values. A stricter production model should use timestamped author-history features or remove these fields.

5. **Dataset time period**  
   The dataset mainly covers Medium activity up to 2018. Medium behavior and platform dynamics may have changed since then.


## **Feature importance and target-encoding ablation**

LightGBM feature importance showed that author/publication history is highly predictive.

Top LightGBM feature groups by gain importance:

| Feature group | Gain importance share |
|---|---:|
| Target encoding | 60.93% |
| Engineered numeric | 18.25% |
| SVD-compressed sparse features | 15.42% |
| Frequency | 5.34% |
| Other | 0.06% |

The strongest individual features included:

- `publication_target_encoding_lgb`
- `author_target_encoding_lgb`
- `usersFollowedByCount_value`
- `svd_sparse_001`
- `publication_frequency_lgb`
- `author_frequency_lgb`

A target-encoding ablation confirmed that these features matter:

| Model | Validation MAE |
|---|---:|
| Final Ridge/LightGBM blend | **1.09060** |
| LightGBM full | 1.12090 |
| Selected Ridge | 1.15361 |
| LightGBM without target encoding | 1.19271 |

Removing target-encoding features increased LightGBM validation MAE by:

```text
0.07181
```

The target encodings are valid in this project because they are built without validation/test target leakage. However, the ablation also shows that the model depends heavily on historical author/publication performance.


## **Leakage prevention and chronological validation**

Leakage prevention is one of the main goals of this project.

The project uses the following safeguards:

- The train/validation/test split is chronological.
- Validation and test articles are later in time than the training articles.
- TF-IDF vectorizers are fitted only on the appropriate training split.
- Categorical vectorizers are fitted only on the appropriate training split.
- Numeric imputers and scalers are fitted only on the appropriate training split.
- SVD models are fitted only on the appropriate training sparse matrix.
- The clipped-target threshold is computed from the training split only.
- Training target encodings use leave-one-out logic.
- Validation target encodings use statistics learned only from the training split.
- Test target encodings use statistics learned only from train + validation during final evaluation.
- Blend weights are selected using validation MAE.
- Mean-shift calibration is selected using validation performance.
- The held-out test set is used only for final reporting and diagnostics.

A stricter production system could go further by using rolling chronological target encodings, especially for author and publication history. The notebook acknowledges this limitation and treats the current target encodings as leakage-free with respect to validation/test targets, not as a complete production-time feature-history system.


## **Saved final outputs**

The notebook saves two final output files:

```text
medium_claps_outputs/final_model_comparison.csv
medium_claps_outputs/final_blend_test_predictions.csv
```

The prediction file includes:

- article ID
- publication timestamp
- title
- author
- publication
- true claps
- true `log_claps`
- predicted `log_claps`
- predicted claps
- log-scale residual
- log-scale absolute error

The raw dataset cache is saved locally as:

```text
medium_claps_outputs/medium_raw_dataset.parquet
```

This file is large so it is excluded from GitHub.


## **Repository structure**

Recommended repository structure:

```text
Medium_Claps_Prediction/
├── README.md
├── medium_claps_prediction.ipynb
├── requirements.txt
├── .gitignore
└── medium_claps_outputs/
    ├── final_model_comparison.csv
    ├── final_blend_test_predictions.csv
    └── medium_raw_dataset.parquet
```


## **How to run the notebook**

1. Clone the repository.

```bash
git clone https://github.com/Nahid-ahmdv/Medium_Claps_Prediction.git
cd Medium_Claps_Prediction
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Launch Jupyter.

```bash
jupyter notebook
```

5. Open and run:

```text
medium_claps_prediction.ipynb
```

On the first run, the notebook downloads the dataset from Hugging Face and saves a local cache. Later runs load from the local Parquet cache unless `FORCE_REDOWNLOAD = True`.

For a faster development run, set:

```python
MAX_ROWS = 100_000
```

For the full reported results, keep:

```python
MAX_ROWS = None
```

## **Requirements and dependencies**

The repository includes a `requirements.txt` file.

Main Python libraries used by the notebook include:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `lightgbm`
- `datasets`
- `pyarrow`

Install them with:

```bash
pip install -r requirements.txt
```

The notebook also includes an optional install cell for missing packages.


## **Key takeaways**

This project demonstrates:

- framing a real-world long-tailed regression problem
- careful target transformation with `log1p(claps)`
- clean data preparation and diagnostics
- chronological validation instead of random splitting
- leakage-aware feature construction
- high-dimensional sparse NLP modeling with Ridge
- clipped-target modeling for rare viral outliers
- nonlinear modeling with LightGBM
- SVD compression of sparse features
- validation-based blending and calibration
- held-out test evaluation
- error analysis and model limitations
- feature importance and ablation testing
- saved final outputs for reproducibility

The final blend is stronger than either individual model because Ridge and LightGBM capture complementary signals.


## **Future improvements**

Possible next steps:

- Use rolling chronological target encoding for stricter production-time author/publication history.
- Replace scrape-time author metadata with timestamped author-history features.
- Add semantic text embeddings such as Sentence-BERT or transformer embeddings.
- Model zero-clap articles separately with a two-stage classifier + regressor.
- Try objectives designed for long-tailed targets, such as quantile loss.
- Tune LightGBM hyperparameters more extensively.
- Evaluate performance separately for cold-start authors and publications.
- Add calibration plots and prediction intervals.
- Test the workflow on a newer Medium dataset if one becomes available.


## **Final note**

The final model is a strong portfolio-quality benchmark for predicting Medium article engagement on the `log1p(claps)` scale. It is not claimed to be production-ready, but it shows a complete and careful machine learning workflow with explicit attention to chronological validation, leakage prevention, model comparison, calibration, interpretation, and limitations.

- Medium article: [soon]
----

⭐ Hope you find this project as instructive and insightful as I did while building it!
