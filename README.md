# House Price Prediction

Regression models that predict `median_house_value` for California housing districts, from XGBoost to a side-by-side comparison of eight scikit-learn regressors.

## Dataset
`housing.csv` (included): the California Housing dataset, with 20,640 districts and 10 columns: location (`longitude`, `latitude`), `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity` (categorical), and the target `median_house_value`.

Both notebooks drop the 207 rows where `total_bedrooms` is missing.

## Notebooks

### `house price pred.ipynb` — XGBoost
- Uses the 8 numeric features, scaled with `StandardScaler` (`ocean_proximity` is not used)
- Baseline: Linear Regression, 5-fold CV MAE of **$52,798**
- `XGBRegressor` (100 trees, max depth 7, learning rate 0.1) on an 80/20 split: **R² 0.831, MAE $31,169**
- Feature importance (by gain), actual-vs-predicted plot, SHAP beeswarm plot, residual plots

### `testing ML Alghrothim.ipynb` — model comparison
- EDA: correlation heatmap, target distribution, `ocean_proximity` counts
- Feature engineering: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, plus one-hot encoded `ocean_proximity`
- Eight models trained on the same 80/20 split (`random_state=42`):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Random Forest | 0.813 | $32,752 | $50,536 |
| Gradient Boosting | 0.785 | $37,203 | $54,189 |
| KNN | 0.726 | $40,822 | $61,264 |
| Decision Tree | 0.664 | $43,006 | $67,770 |
| Ridge / Linear / Lasso | 0.651 | $49,774 | $69,043 |
| SVR (RBF, C=10) | 0.013 | $85,964 | $116,192 |

- Random Forest tuned with `RandomizedSearchCV` (20 iterations, 5-fold): **R² 0.819, RMSE $49,742**
- Feature-importance chart for the tuned model

## Installation
```bash
git clone https://github.com/Ahmed-309/house-price-pred.git
cd house-price-pred
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
```

## Usage
```bash
jupyter notebook
```
Then open either notebook.

- `house price pred.ipynb` reads `housing.csv` from the repo folder, so it works as-is. Cell 12 (`preds.shape`) refers to a variable that is never defined; skip or delete it when running all cells.
- `testing ML Alghrothim.ipynb` reads from `D:\AI\poly\housing.csv`. Change it to `pd.read_csv("housing.csv")` before running.

## Project Structure
| File | Purpose |
|---|---|
| `house price pred.ipynb` | XGBoost model with SHAP and residual analysis |
| `testing ML Alghrothim.ipynb` | EDA, feature engineering, and 8-model comparison with tuning |
| `housing.csv` | California Housing dataset |

## Requirements
- Python 3.10
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
