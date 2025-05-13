import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
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
    "LDAU", "H1_ns", "H5_ns", "Unnamed: 97"
]

# === FEATURE ENGINEERING FUNCTION ===
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

# Use one genre column and drop the rest
df["genre"] = df["genre_x"]
df = df.drop(columns=["genre_x", "genre_y"], errors='ignore')

# Drop excluded columns
df = df.drop(columns=[col for col in EXCLUDE_COLS if col in df.columns], errors='ignore')

# Define features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != "H1_all"]

# Fill missing values with 0 instead of dropping
df = df[feature_cols + ["H1_all", "show_id", "genre"]].fillna(0)

# Add log features
df = add_engineered_features(df)
all_features = [col for col in df.columns if col not in ['H1_all', 'show_id', 'genre']]
df = df[all_features + ['H1_all', 'show_id', 'genre']].fillna(0)

# Bin H1_all for stratified sampling
df["H1_bin"] = pd.qcut(df["H1_all"], q=3, duplicates='drop', labels=False)

# === RESULT CONTAINERS ===
results = []
metrics_list = []

print(f"\nGenres in dataset: {df['genre'].unique().tolist()}")

# === PER GENRE MODELING ===
for genre, genre_df in df.groupby("genre"):
    print(f"\n🔍 Processing genre: {genre}")
    print(f"  Total rows: {len(genre_df)}")

    bin_counts = genre_df["H1_bin"].value_counts().sort_index()
    print(f"  Bin counts:\n{bin_counts.to_string()}")

    use_stratified = True
    if bin_counts.min() < 2 or len(genre_df) < 10:
        print(f"  ⚠️ Using random split (stratified not feasible)")
        use_stratified = False

    X = genre_df[all_features]
    y = genre_df["H1_all"]

    if use_stratified:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        split_gen = splitter.split(X, genre_df["H1_bin"])
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        split_gen = splitter.split(X, y)

    for train_idx, test_idx in split_gen:
        train_df = genre_df.iloc[train_idx]
        test_df = genre_df.iloc[test_idx]

        X_train = train_df[all_features]
        y_train = train_df["H1_all"]
        X_test = test_df[all_features].copy()
        y_test = test_df["H1_all"]

        # === Remove H5_all signals from test set
        h5_cols = [col for col in X_test.columns if 'H5_all' in col]
        for h5_col in h5_cols:
            X_test[h5_col] = 0

        # Train model on log scale
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

        # Predict and back-transform
        raw_preds_log = model.predict(X_test)
        raw_preds = np.expm1(raw_preds_log)

        train_preds_log = model.predict(X_train)
        train_preds = np.expm1(train_preds_log)

        # Z-score calibration
        mean_pred = np.mean(train_preds)
        std_pred = np.std(train_preds)
        mean_target = np.mean(y_train)
        std_target = np.std(y_train)

        z_scores = (raw_preds - mean_pred) / std_pred
        adjusted_preds = mean_target + z_scores * std_target

        # Store predictions
        temp_result = pd.DataFrame({
            "show_id": test_df["show_id"].values,
            "genre": test_df["genre"].values,
            "predicted_retention": adjusted_preds
        })
        results.append(temp_result)

        # Metrics
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

# === SAVE RESULTS ===
if results:
    final_predictions = pd.concat(results, ignore_index=True)
    final_predictions.to_excel(output_path, index=False)
    print(f"\n✅ Final retention predictions saved to: {output_path}")
else:
    print("\n❌ No predictions generated. All genres may have been skipped.")

# === PRINT METRICS ===
print("\n📊 Second: What Metrics to Track\n")
if metrics_list:
    for metrics in metrics_list:
        print(f"Genre: {metrics['genre']}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R² Score: {metrics['R2_Score']:.4f}")
        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%\n")
else:
    print("❌ No metrics to report. All genre splits were skipped.")
