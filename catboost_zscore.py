import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# === FILE PATHS ===
input_path = "/Users/shubhamsharma/Desktop/content_appeal_all_shows.xlsx"
output_path = "/Users/shubhamsharma/Desktop/final_predicted_retention.xlsx"

# === EXCLUDE COLUMNS ===
EXCLUDE_COLS = [
    "word_count_x", "word_count_y", "total_word_count",
    "character_count_with_spaces", "character_count_without_spaces",
    "LDAU", "H1_ns", "H5_ns"
]

# === FEATURE ENGINEERING FUNCTION (Safe Version) ===
def add_engineered_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[f"log_{col}"] = np.log1p(df[col])
    return df

# === LOAD AND PREPARE DATA ===
sheet1 = pd.read_excel(input_path, sheet_name=0)
sheet2 = pd.read_excel(input_path, sheet_name=1)
sheet1.columns = sheet1.columns.str.strip()
sheet2.columns = sheet2.columns.str.strip()
df = pd.merge(sheet1, sheet2, on="show_id", how="inner")

df = df.drop(columns=[col for col in EXCLUDE_COLS if col in df.columns], errors='ignore')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != "H1_all"]
df = df[feature_cols + ["H1_all", "show_id", "genre"]].dropna()
df = add_engineered_features(df)
all_features = [col for col in df.columns if col not in ['H1_all', 'show_id', 'genre']]
df = df[all_features + ['H1_all', 'show_id', 'genre']].dropna()
df["H1_bin"] = pd.qcut(df["H1_all"], q=10, duplicates='drop', labels=False)

# === RESULT CONTAINERS ===
results = []
metrics_list = []

# === PER GENRE MODELING ===
for genre, genre_df in df.groupby("genre"):
    X = genre_df[all_features]
    y = genre_df["H1_all"]
    bins = genre_df["H1_bin"]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, bins):
        train_df = genre_df.iloc[train_idx]
        test_df = genre_df.iloc[test_idx]

        X_train = train_df[all_features]
        y_train = train_df["H1_all"]
        X_test_full = test_df[all_features]
        y_test = test_df["H1_all"]

        X_test = X_test_full.copy()
        h5_cols = [col for col in X_test.columns if 'H5_all' in col]
        for h5_col in h5_cols:
            X_test[h5_col] = 0

        y_train_log = np.log1p(y_train)

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.02,
            depth=9,
            loss_function='Quantile:alpha=0.5',
            verbose=100,
            random_state=42
        )
        model.fit(X_train, y_train_log)

        raw_preds_log = model.predict(X_test)
        raw_preds = np.expm1(raw_preds_log)

        train_preds_log = model.predict(X_train)
        train_preds = np.expm1(train_preds_log)

        mean_pred = np.mean(train_preds)
        std_pred = np.std(train_preds)
        mean_target = np.mean(y_train)
        std_target = np.std(y_train)

        z_scores = (raw_preds - mean_pred) / std_pred
        adjusted_preds = mean_target + z_scores * std_target

        temp_result = pd.DataFrame({
            "show_id": test_df["show_id"].values,
            "genre": test_df["genre"].values,
            "predicted_retention": adjusted_preds
        })
        results.append(temp_result)

        mae = mean_absolute_error(y_test, adjusted_preds)
        rmse = np.sqrt(mean_squared_error(y_test, adjusted_preds))
        r2 = r2_score(y_test, adjusted_preds)
        mape = np.mean(np.abs((y_test - adjusted_preds) / y_test)) * 100

        metrics_list.append({
            "genre": genre,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2,
            "MAPE (%)": mape
        })

# === CONCAT AND SAVE ===
final_predictions = pd.concat(results, ignore_index=True)
final_predictions.to_excel(output_path, index=False)

print("\nSecond: What Metrics to Track\n")
for metrics in metrics_list:
    print(f"Genre: {metrics['genre']}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R² Score: {metrics['R2_Score']:.4f}")
    print(f"  MAPE: {metrics['MAPE (%)']:.2f}%\n")

print(f"\n✅ Final retention predictions saved to: {output_path}")
