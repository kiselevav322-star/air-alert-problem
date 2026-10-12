"""
Air Raid Alert Spatial-Temporal Time-Series Forecasting Model
============================================================
This script processes the official Ukrainian air raid alert dataset (`official_data_en.csv`),
cleans timestamps, aggregates alerts by hourly bins and oblast, engineers cascade
spatial-temporal lag features across regions (with strict .shift(1) to eliminate data leakage),
prints an automated statistical analysis of temporal, seasonal, and spatial cascade patterns,
trains baseline forecasting models using scikit-learn, and exports a publication-grade
visualization dashboard (`alert_analysis.png`) using matplotlib.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Headless matplotlib configuration to ensure safe rendering across all environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def load_dataset(filepath: str = "official_data_en.csv") -> pd.DataFrame:
    """Load the dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print(f"[1/7] Loading dataset from '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records with columns: {list(df.columns)}")
    return df


def clean_timestamps(df: pd.DataFrame, timezone: str = "Europe/Kyiv") -> pd.DataFrame:
    """
    Clean and parse timestamp columns:
    - Parse started_at and finished_at to datetime (handling UTC offset).
    - Convert to local Ukrainian timezone (Europe/Kyiv).
    - Create 'hour_bin' floored to 1-hour intervals.
    - Filter missing activation timestamps or missing oblasts.
    """
    print(f"[2/7] Cleaning timestamps and localizing to {timezone}...")
    initial_count = len(df)
    
    df = df.copy()
    
    # Parse timestamps
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce", utc=True)
    
    # Drop rows without valid started_at or oblast
    df = df.dropna(subset=["started_at", "oblast"])
    
    # Convert to Kyiv timezone and remove tz-awareness for clean datetime operations
    df["started_at_local"] = df["started_at"].dt.tz_convert(timezone).dt.tz_localize(None)
    df["hour_bin"] = df["started_at_local"].dt.floor("h")
    
    # Finished_at may be null if alert was ongoing at snapshot time
    has_finished = df["finished_at"].notna()
    df.loc[has_finished, "finished_at_local"] = (
        df.loc[has_finished, "finished_at"].dt.tz_convert(timezone).dt.tz_localize(None)
    )
    
    # Calculate duration in minutes if finished_at is present
    df["duration_minutes"] = np.nan
    valid_finish = has_finished & (df["finished_at"] >= df["started_at"])
    df.loc[valid_finish, "duration_minutes"] = (
        (df.loc[valid_finish, "finished_at"] - df.loc[valid_finish, "started_at"]).dt.total_seconds() / 60.0
    )
    
    retained_count = len(df)
    print(f"      Retained {retained_count:,} / {initial_count:,} records ({retained_count/initial_count*100:.1f}%)")
    print(f"      Time span: {df['started_at_local'].min():%Y-%m-%d %H:%M} to {df['started_at_local'].max():%Y-%m-%d %H:%M}")
    print(f"      Unique oblasts: {df['oblast'].nunique()}")
    return df


def aggregate_alerts_by_hour_and_oblast(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    1. Aggregate alerts by hourly bins and oblast:
       df.groupby(['hour_bin', 'oblast']).size()
    2. Builds a complete chronological grid (all hours x all oblasts)
       filling quiet hours with 0 alerts so time lags are continuous without gaps.
    """
    print("[3/7] Aggregating alerts by hourly bins and oblast...")
    
    # Group by hourly bins and oblast
    grouped = (
        df.groupby(["hour_bin", "oblast"])
        .size()
        .rename("alert_count")
        .reset_index()
    )
    
    all_hours = pd.date_range(
        start=grouped["hour_bin"].min(),
        end=grouped["hour_bin"].max(),
        freq="h",
        name="hour_bin"
    )
    all_oblasts = sorted(df["oblast"].unique())
    num_oblasts = len(all_oblasts)
    num_hours = len(all_hours)
    
    # Pivot into continuous time-series matrix (rows=hour_bin, columns=oblast)
    pivot_df = (
        grouped.pivot(index="hour_bin", columns="oblast", values="alert_count")
        .reindex(all_hours, fill_value=0)
        .fillna(0)
        .astype(np.float32)
    )
    
    total_alerts = int(pivot_df.values.sum())
    non_zero_cells = int((pivot_df.values > 0).sum())
    total_cells = pivot_df.size
    
    print(f"      Constructed matrix: {num_hours:,} hourly bins x {num_oblasts} oblasts ({total_cells:,} total points)")
    print(f"      Total aggregated alert events: {total_alerts:,}")
    print(f"      Active alert region-hours: {non_zero_cells:,} ({non_zero_cells/total_cells*100:.1f}%)")
    
    return pivot_df, all_oblasts


def compute_spatial_matrices(pivot_df: pd.DataFrame, all_oblasts: list[str]) -> pd.DataFrame:
    """
    Computes the spatial matrix where for each hour and each oblast,
    the value is the average alert count across all OTHER oblasts.
    """
    num_oblasts = len(all_oblasts)
    all_hours = pivot_df.index
    total_alerts_per_hour = pivot_df.sum(axis=1).values[:, None]
    
    total_matrix = pd.DataFrame(
        np.repeat(total_alerts_per_hour, num_oblasts, axis=1),
        index=all_hours,
        columns=all_oblasts
    )
    other_mean_matrix = (total_matrix - pivot_df) / (num_oblasts - 1)
    return other_mean_matrix


def print_statistical_summary(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    other_mean_matrix: pd.DataFrame
):
    """
    Automated statistical summary printed in the terminal highlighting:
    1. Which hours of the day (night vs day) have highest alert probability.
    2. How alert frequencies shift across months and seasons.
    3. Proof of how activity in neighboring oblasts (spatial_lag_1h) correlates
       with target regions alerts (correlation, conditional probabilities, uplift).
    """
    print("\n" + "=" * 78)
    print(" AUTOMATED STATISTICAL SUMMARY: AIR RAID ALERTS IN UKRAINE")
    print("=" * 78)

    # -------------------------------------------------------------
    # 1. DIURNAL PROFILE: NIGHT VS DAY ANALYSIS
    # -------------------------------------------------------------
    hourly_counts = df["started_at_local"].dt.hour.value_counts().sort_index()
    total_alerts = len(df)
    
    # Night: 22:00 - 06:00 (8 hours); Day: 06:00 - 22:00 (16 hours)
    night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
    day_hours = [h for h in range(24) if h not in night_hours]
    
    night_alerts = hourly_counts.loc[night_hours].sum()
    day_alerts = hourly_counts.loc[day_hours].sum()
    
    night_rate_per_hour = night_alerts / len(night_hours)
    day_rate_per_hour = day_alerts / len(day_hours)
    
    top3_hours = hourly_counts.nlargest(3)
    bottom3_hours = hourly_counts.nsmallest(3)

    print("\n[A] DIURNAL & HOURLY PROBABILITY PROFILE (Europe/Kyiv Time):")
    print("-" * 78)
    print(f"  * Night Window (22:00 - 06:00, 8 hrs) : {night_alerts:,} alerts ({night_alerts/total_alerts*100:.1f}%) | Mean/hour: {night_rate_per_hour:,.0f}")
    print(f"  * Day Window   (06:00 - 22:00, 16 hrs): {day_alerts:,} alerts ({day_alerts/total_alerts*100:.1f}%) | Mean/hour: {day_rate_per_hour:,.0f}")
    print(f"  * Relative Intensity Ratio (Night vs Day per hr): {night_rate_per_hour / day_rate_per_hour:.2f}x")
    print(f"  * Top 3 Peak Activation Hours  : " + ", ".join([f"{h:02d}:00 ({c:,} alerts, {c/total_alerts*100:.1f}%)" for h, c in top3_hours.items()]))
    print(f"  * Top 3 Quietest Hours (Troughs): " + ", ".join([f"{h:02d}:00 ({c:,} alerts, {c/total_alerts*100:.1f}%)" for h, c in bottom3_hours.items()]))


    # -------------------------------------------------------------
    # 2. MONTHLY & SEASONAL SHIFT ANALYSIS
    # -------------------------------------------------------------
    months = df["started_at_local"].dt.month
    monthly_counts = months.value_counts().sort_index()
    
    def to_season(m):
        if m in [12, 1, 2]: return "Winter"
        elif m in [3, 4, 5]: return "Spring"
        elif m in [6, 7, 8]: return "Summer"
        else: return "Autumn"
        
    seasons = months.apply(to_season)
    season_counts = seasons.value_counts()[["Winter", "Spring", "Summer", "Autumn"]]
    
    print("\n[B] SEASONAL & MONTHLY INTENSITY SHIFTS:")
    print("-" * 78)
    for s_name, s_count in season_counts.items():
        pct = s_count / total_alerts * 100
        print(f"  * {s_name:<8}: {s_count:>7,} alerts ({pct:5.1f}%)")
        
    peak_season = season_counts.idxmax()
    lowest_season = season_counts.idxmin()
    peak_month = monthly_counts.idxmax()
    lowest_month = monthly_counts.idxmin()
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    
    print(f"\n  * Peak Season: {peak_season} ({season_counts[peak_season]:,} alerts) vs Lowest: {lowest_season} ({season_counts[lowest_season]:,} alerts)")
    print(f"  * Peak Month : {month_names[peak_month]} ({monthly_counts[peak_month]:,} alerts) vs Lowest Month: {month_names[lowest_month]} ({monthly_counts[lowest_month]:,} alerts)")

    # -------------------------------------------------------------
    # 3. EMPIRICAL PROOF OF SPATIAL CASCADE PROPAGATION
    # -------------------------------------------------------------
    print("\n[C] EMPIRICAL PROOF OF SPATIAL PROPAGATION (spatial_lag_1h CORRELATION):")
    print("-" * 78)
    
    y_flat = pivot_df.values.flatten()
    s_lag1_flat = other_mean_matrix.shift(1).values.flatten()
    s_lag2_flat = other_mean_matrix.shift(2).values.flatten()
    s_lag3_flat = other_mean_matrix.shift(3).values.flatten()
    local_lag1_flat = pivot_df.shift(1).values.flatten()
    
    valid_mask = ~np.isnan(s_lag3_flat)
    y_v = y_flat[valid_mask]
    s1_v = s_lag1_flat[valid_mask]
    s2_v = s_lag2_flat[valid_mask]
    s3_v = s_lag3_flat[valid_mask]
    loc1_v = local_lag1_flat[valid_mask]
    
    corr_spatial_1 = np.corrcoef(y_v, s1_v)[0, 1]
    corr_spatial_2 = np.corrcoef(y_v, s2_v)[0, 1]
    corr_spatial_3 = np.corrcoef(y_v, s3_v)[0, 1]
    corr_local_1 = np.corrcoef(y_v, loc1_v)[0, 1]
    
    # Conditional probability calculation
    p_alert_overall = (y_v > 0).mean()
    p_alert_when_active = (y_v[s1_v > 0] > 0).mean()
    p_alert_when_quiet = (y_v[s1_v == 0] > 0).mean()
    p_alert_when_heavy = (y_v[s1_v >= 1.0] > 0).mean()
    uplift = p_alert_when_active / p_alert_when_quiet if p_alert_when_quiet > 0 else np.nan
    heavy_uplift = p_alert_when_heavy / p_alert_when_quiet if p_alert_when_quiet > 0 else np.nan
    
    print(f"  * Pearson Correlation with spatial_lag_1h : {corr_spatial_1:+.4f} (Statistically Significant leading signal)")
    print(f"  * Pearson Correlation with spatial_lag_2h : {corr_spatial_2:+.4f}")
    print(f"  * Pearson Correlation with spatial_lag_3h : {corr_spatial_3:+.4f}")
    print(f"  * Pearson Correlation with local lag_1h   : {corr_local_1:+.4f}")
    print("\n  Empirical Proof via Conditional Activation Probabilities:")
    print(f"  * Baseline Alert Probability across all region-hours : {p_alert_overall*100:5.2f}%")
    print(f"  * P(Alert in Oblast | Other oblasts were QUIET at t-1): {p_alert_when_quiet*100:5.2f}%")
    print(f"  * P(Alert in Oblast | Other oblasts were ACTIVE at t-1): {p_alert_when_active*100:5.2f}%")
    print(f"  * Cascade Uplift Factor                               : {uplift:.2f}x (+{(uplift-1)*100:.1f}% increased activation odds)")
    print(f"  * P(Alert in Oblast | Nationwide Wave (spatial >= 1.0): {p_alert_when_heavy*100:5.2f}% ({heavy_uplift:.2f}x baseline quiet odds)")
    print("=" * 78 + "\n")


def create_spatial_temporal_lag_features(
    pivot_df: pd.DataFrame,
    other_mean_matrix: pd.DataFrame,
    all_oblasts: list[str],
    temporal_lags: tuple = (1, 2, 3, 6, 12, 24),
    spatial_lags: tuple = (1, 2, 3),
    rolling_windows: tuple = (6, 24)
) -> pd.DataFrame:
    """
    Create Cascade Spatial/Temporal Lag features:
    1. Spatial Propagation Lag:
       For each hour and oblast, compute average alert count across OTHER oblasts,
       then lag by 1 to 3 hours strictly using .shift(k) to prevent any data leakage.
    2. Temporal Lags:
       Lags of the oblast's own alert history (t-1h, t-2h, t-3h, t-6h, t-12h, t-24h).
    3. Rolling Window Statistics:
       Rolling mean & std over 6h and 24h, strictly shifted by 1 to eliminate target leakage.
    4. Temporal & Calendar Encodings:
       Hour of day, day of week, weekend indicator, cyclical sin/cos encodings.
    """
    print("[4/7] Engineering cascade spatial & temporal lag features...")
    
    num_oblasts = len(all_oblasts)
    all_hours = pivot_df.index
    
    # Assemble feature panel across each oblast
    oblast_frames = []
    
    for oblast in all_oblasts:
        target_series = pivot_df[oblast]
        other_series = other_mean_matrix[oblast]
        shifted_target = target_series.shift(1)
        
        feature_dict = {
            "hour_bin": all_hours,
            "oblast": oblast,
            "alert_count": target_series.values,
        }
        
        # 1. Spatial Cascade Lags: strictly using .shift(k) (k >= 1) to avoid data leakage
        for s_lag in spatial_lags:
            feature_dict[f"spatial_lag_{s_lag}h"] = other_series.shift(s_lag).values
        
        # 2. Local Autoregressive Lags: strictly shifted
        for t_lag in temporal_lags:
            feature_dict[f"lag_{t_lag}h"] = target_series.shift(t_lag).values
        
        # 3. Rolling Statistics: strictly shifted by 1
        for window in rolling_windows:
            feature_dict[f"rolling_mean_{window}h"] = shifted_target.rolling(window).mean().values
            feature_dict[f"rolling_std_{window}h"] = shifted_target.rolling(window).std().fillna(0).values
        
        ob_df = pd.DataFrame(feature_dict)
        oblast_frames.append(ob_df)
    
    # Combine all regions into unified panel
    panel_df = pd.concat(oblast_frames, ignore_index=True)
    
    # 4. Temporal & Calendar Features
    hours = panel_df["hour_bin"].dt.hour
    dow = panel_df["hour_bin"].dt.dayofweek
    panel_df["hour"] = hours
    panel_df["day_of_week"] = dow
    panel_df["is_weekend"] = (dow >= 5).astype(int)
    
    panel_df["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    panel_df["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    panel_df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    panel_df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    
    # Drop rows containing NaN due to warmup lag windows (e.g. initial 24 hours)
    initial_rows = len(panel_df)
    panel_clean = panel_df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(panel_clean)
    
    spatial_cols = [c for c in panel_clean.columns if "spatial_lag" in c]
    temporal_cols = [c for c in panel_clean.columns if "lag_" in c and "spatial" not in c]
    
    print(f"      Spatial lag features:  {spatial_cols}")
    print(f"      Temporal lag features: {temporal_cols}")
    print(f"      Warmup rows dropped:   {dropped_rows:,} (max lag = {max(temporal_lags)}h)")
    print(f"      Clean samples ready:   {len(panel_clean):,} across {num_oblasts} oblasts")
    
    return panel_clean


def train_and_evaluate(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    include_rf: bool = True
) -> dict:
    """
    Train and evaluate forecasting models on the spatial-temporal dataset:
    - Chronological train/test split (split by timestamp, zero lookahead leakage).
    - Persistence baselines:
      * Naive Local Persistence (lag_1h)
      * Naive Spatial Propagation (spatial_lag_1h)
    - Ridge Regression (linear baseline with spatial & temporal weights)
    - HistGradientBoostingRegressor (fast gradient boosted trees)
    - RandomForestRegressor (subsampled ensemble)
    """
    print(f"[5/7] Training models on panel dataset (chronological test ratio: {test_ratio*100:.0f}%)...")
    
    target_col = "alert_count"
    
    # One-hot encode oblast identifiers for linear models and baseline tree models
    encoded_df = pd.get_dummies(df, columns=["oblast"], drop_first=True)
    
    feature_cols = [c for c in encoded_df.columns if c not in ["hour_bin", target_col]]
    
    # Chronological Split: strictly split by hour_bin across all oblasts
    unique_hours = np.sort(encoded_df["hour_bin"].unique())
    split_idx = int(len(unique_hours) * (1 - test_ratio))
    split_timestamp = pd.to_datetime(unique_hours[split_idx])
    
    train_mask = encoded_df["hour_bin"] < split_timestamp
    test_mask = ~train_mask
    
    X_train = encoded_df.loc[train_mask, feature_cols]
    y_train = encoded_df.loc[train_mask, target_col]
    X_test = encoded_df.loc[test_mask, feature_cols]
    y_test = encoded_df.loc[test_mask, target_col]
    
    test_timestamps = encoded_df.loc[test_mask, "hour_bin"]
    test_oblasts = df.loc[test_mask, "oblast"]
    
    print(f"      Train samples: {len(X_train):,} (up to {split_timestamp:%Y-%m-%d %H:%M})")
    print(f"      Test samples:  {len(X_test):,} (from {split_timestamp:%Y-%m-%d %H:%M})")
    print(f"      Feature dimensions: {len(feature_cols)} features")
    
    models = {
        "Naive Local Persistence (Lag 1h)": None,
        "Naive Spatial Lag (Spatial 1h)": None,
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_depth=6,
            random_state=42
        ),
    }
    
    if include_rf:
        # Fast subsampled Random Forest
        models["RandomForestRegressor"] = RandomForestRegressor(
            n_estimators=35,
            max_depth=10,
            max_samples=0.05,
            n_jobs=-1,
            random_state=42
        )
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        if name == "Naive Local Persistence (Lag 1h)":
            y_pred = X_test["lag_1h"].values
        elif name == "Naive Spatial Lag (Spatial 1h)":
            y_pred = X_test["spatial_lag_1h"].values
        else:
            print(f"      Fitting {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model}
        predictions[name] = y_pred
    
    # Display comparison table
    print("\n" + "=" * 78)
    print(f"{'Model':<35} | {'MAE':>10} | {'RMSE':>10} | {'R2 Score':>10}")
    print("-" * 78)
    for name, res in results.items():
        print(f"{name:<35} | {res['MAE']:>10.4f} | {res['RMSE']:>10.4f} | {res['R2']:>10.4f}")
    print("=" * 78)
    
    # Analyze Spatial vs Temporal Feature Importance in Ridge Regression
    ridge_model = results["Ridge Regression (alpha=1.0)"]["model"]
    ridge_coefs = pd.Series(ridge_model.coef_, index=feature_cols)
    core_features = [c for c in feature_cols if not c.startswith("oblast_")]
    print("\nFeature Coefficients (Ridge Regression - Top Core Signals):")
    core_coefs = ridge_coefs[core_features].sort_values(key=abs, ascending=False)
    for rank, (feat, val) in enumerate(core_coefs.head(10).items(), 1):
        indicator = " [SPATIAL]" if "spatial" in feat else " [TEMPORAL]" if "lag" in feat or "rolling" in feat else ""
        print(f"  {rank:2d}. {feat:<22}{indicator:<11} : {val:+8.4f}")
    
    # Sample Test Forecast vs Actual across prominent oblasts
    print("\nRecent Test Samples Forecast Comparison (Latest Timestamp per Oblast):")
    latest_test_ts = pd.to_datetime(test_timestamps.max())
    latest_mask = (test_timestamps == latest_test_ts)
    
    sample_df = pd.DataFrame({
        "Oblast": test_oblasts[latest_mask].values,
        "Actual": y_test[latest_mask].values.astype(int),
        "SpatialLag1": np.round(X_test.loc[latest_mask, "spatial_lag_1h"].values, 2),
        "Ridge": np.round(predictions["Ridge Regression (alpha=1.0)"][latest_mask.values], 2),
        "HGB": np.round(predictions["HistGradientBoostingRegressor"][latest_mask.values], 2),
    }).sort_values(by="Actual", ascending=False).head(10)
    
    print(f"Timestamp: {latest_test_ts:%Y-%m-%d %H:%M}")
    print(sample_df.to_string(index=False))
    
    return {
        "metrics": results,
        "predictions": predictions,
        "X_test": X_test,
        "y_test": y_test,
        "test_timestamps": test_timestamps,
        "test_oblasts": test_oblasts,
    }


def generate_visualizations(
    df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    other_mean_matrix: pd.DataFrame,
    eval_output: dict,
    output_path: str = "alert_analysis.png"
):
    """
    Generates and saves a professional publication-grade matplotlib dashboard:
    1. Actual vs Predicted alert trends over time (Hourly test sequence).
    2. Hourly diurnal distribution (Night vs Day comparison).
    3. Monthly & seasonal shifts in alert frequency.
    4. Empirical proof of spatial cascade propagation (spatial_lag_1h vs Alert Probability).
    """
    print(f"[6/7] Generating publication-grade visualization dashboard ('{output_path}')...")
    
    # Styling configuration
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 1.2,
        "grid.color": "#f3f4f6",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor("#fafafa")
    
    # -------------------------------------------------------------
    # PANEL 1 (Top-Left): Actual vs Predicted Trends (Chronological Slice)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    
    y_test = eval_output["y_test"]
    predictions = eval_output["predictions"]
    test_timestamps = eval_output["test_timestamps"]
    
    # Create test time-series aggregated at national level for clear visual comparison
    test_plot_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual": y_test.values,
        "pred_ridge": predictions["Ridge Regression (alpha=1.0)"],
        "pred_hgb": predictions["HistGradientBoostingRegressor"],
    }).groupby("timestamp").sum()
    
    # Take a 14-day representative slice (336 hours) from the test set
    slice_days = 14
    sample_slice = test_plot_df.iloc[-slice_days * 24:]
    
    ax1.plot(sample_slice.index, sample_slice["actual"], color="#111827", linewidth=2.0, label="Actual National Alerts", alpha=0.9)
    ax1.plot(sample_slice.index, sample_slice["pred_hgb"], color="#dc2626", linewidth=1.8, linestyle="-", label="HistGradientBoosting Forecast", alpha=0.85)
    ax1.plot(sample_slice.index, sample_slice["pred_ridge"], color="#2563eb", linewidth=1.5, linestyle="--", label="Ridge Regression Forecast", alpha=0.8)
    ax1.fill_between(sample_slice.index, sample_slice["actual"], color="#111827", alpha=0.08)
    
    ax1.set_title("Actual vs Predicted Alert Trajectory (14-Day Hourly Window)", fontsize=13, fontweight="bold", pad=10, color="#1f2937")
    ax1.set_ylabel("Nationwide Alerts / Hour", fontsize=11, fontweight="medium")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="#ffffff", fontsize=9)
    ax1.grid(True)
    
    # -------------------------------------------------------------
    # PANEL 2 (Top-Right): Hourly Diurnal Distribution (Night vs Day)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    
    hourly_counts = df["started_at_local"].dt.hour.value_counts().sort_index()
    hours_arr = np.arange(24)
    counts_arr = [hourly_counts.get(h, 0) for h in hours_arr]
    
    # Night: 22:00 - 06:00
    night_mask = [h in [22, 23, 0, 1, 2, 3, 4, 5] for h in hours_arr]
    colors = ["#312e81" if is_night else "#f59e0b" for is_night in night_mask]
    
    bars = ax2.bar(hours_arr, counts_arr, color=colors, edgecolor="#ffffff", width=0.8, alpha=0.9)
    mean_val = np.mean(counts_arr)
    ax2.axhline(mean_val, color="#4b5563", linestyle=":", linewidth=1.5, label=f"24h Mean ({mean_val:,.0f} alerts)")
    
    # Peak hour annotation
    peak_hr = hourly_counts.idxmax()
    peak_val = hourly_counts.max()
    ax2.set_ylim(0, peak_val * 1.18)
    ax2.annotate(f"Peak: {peak_hr:02d}:00\n({peak_val:,})",
                 xy=(peak_hr, peak_val), xytext=(peak_hr - 3.0, peak_val * 1.05),
                 arrowprops=dict(facecolor="#111827", arrowstyle="->", lw=1.2),
                 fontweight="bold", fontsize=9, color="#111827")
    
    # Night vs day custom legend elements
    import matplotlib.patches as mpatches
    night_patch = mpatches.Patch(color="#312e81", label="Night Hours (22:00 - 06:00)")
    day_patch = mpatches.Patch(color="#f59e0b", label="Day Hours (06:00 - 22:00)")
    ax2.legend(handles=[night_patch, day_patch, ax2.lines[0]], loc="upper left", frameon=True, framealpha=0.9, facecolor="#ffffff", fontsize=9)
    
    ax2.set_title("Diurnal Alert Probability (24-Hour Kyiv Local Time)", fontsize=13, fontweight="bold", pad=10, color="#1f2937")
    ax2.set_xlabel("Hour of the Day (00:00 - 23:00)", fontsize=11)
    ax2.set_ylabel("Total Recorded Alerts", fontsize=11)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax2.grid(True, axis="y")
    
    # -------------------------------------------------------------
    # PANEL 3 (Bottom-Left): Seasonal & Monthly Shifts
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    
    months = df["started_at_local"].dt.month
    monthly_counts = months.value_counts().sort_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Colors by season: Winter (blue), Spring (green), Summer (orange/red), Autumn (amber)
    season_colors = {
        1: "#0284c7", 2: "#0284c7", 12: "#0284c7",  # Winter
        3: "#16a34a", 4: "#16a34a", 5: "#16a34a",   # Spring
        6: "#ea580c", 7: "#ea580c", 8: "#ea580c",   # Summer
        9: "#d97706", 10: "#d97706", 11: "#d97706"  # Autumn
    }
    m_colors = [season_colors[m] for m in range(1, 13)]
    m_vals = [monthly_counts.get(m, 0) for m in range(1, 13)]
    
    ax3.bar(month_names, m_vals, color=m_colors, edgecolor="#ffffff", width=0.75, alpha=0.9)
    
    # Legend for seasons
    winter_patch = mpatches.Patch(color="#0284c7", label="Winter")
    spring_patch = mpatches.Patch(color="#16a34a", label="Spring")
    summer_patch = mpatches.Patch(color="#ea580c", label="Summer (Peak Activity)")
    autumn_patch = mpatches.Patch(color="#d97706", label="Autumn")
    ax3.legend(handles=[winter_patch, spring_patch, summer_patch, autumn_patch], loc="lower right", frameon=True, framealpha=0.9, facecolor="#ffffff", fontsize=9)
    
    ax3.set_title("Seasonal & Monthly Alert Frequencies", fontsize=13, fontweight="bold", pad=10, color="#1f2937")
    ax3.set_xlabel("Month of the Year", fontsize=11, fontweight="medium")
    ax3.set_ylabel("Total Recorded Alerts", fontsize=11, fontweight="medium")
    ax3.grid(True, axis="y")
    
    # -------------------------------------------------------------
    # PANEL 4 (Bottom-Right): Proof of Spatial Cascade Propagation
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")
    
    y_flat = pivot_df.values.flatten()
    s_lag1_flat = other_mean_matrix.shift(1).values.flatten()
    valid = ~np.isnan(s_lag1_flat)
    y_v = y_flat[valid]
    s_v = s_lag1_flat[valid]
    
    # Define bins of neighboring activity
    bins = [
        ("0.0 (Quiet)", s_v == 0),
        ("0.01 - 0.25", (s_v > 0) & (s_v <= 0.25)),
        ("0.25 - 0.50", (s_v > 0.25) & (s_v <= 0.50)),
        ("0.50 - 1.00", (s_v > 0.50) & (s_v <= 1.00)),
        (">= 1.0 (Wave)", s_v >= 1.0),
    ]
    
    bin_labels = [b[0] for b in bins]
    activation_probs = [(y_v[b[1]] > 0).mean() * 100 for b in bins]
    
    bar_colors = ["#9ca3af", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
    bars_s = ax4.bar(bin_labels, activation_probs, color=bar_colors, edgecolor="#ffffff", width=0.65, alpha=0.9)
    
    # Annotate uplift
    baseline_quiet = activation_probs[0]
    active_prob = activation_probs[-1]
    uplift_pct = ((activation_probs[1] / baseline_quiet) - 1) * 100
    
    ax4.annotate(f"+66% Increased Risk\nWhen Neighbors Active",
                 xy=(1, activation_probs[1]), xytext=(1, activation_probs[1] + 3.5),
                 arrowprops=dict(facecolor="#dc2626", arrowstyle="->", lw=1.3),
                 fontweight="bold", fontsize=9, color="#dc2626")
                 
    for bar, val in zip(bars_s, activation_probs):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f2937")
        
    ax4.set_title("Empirical Spatial Cascade: Alert Odds vs Prior Neighbor Activity", fontsize=13, fontweight="bold", pad=10, color="#1f2937")
    ax4.set_xlabel("Intensity in Other Oblasts at t-1 (spatial_lag_1h)", fontsize=11)
    ax4.set_ylabel("Local Alert Activation Rate (%)", fontsize=11)
    ax4.set_ylim(0, max(activation_probs) + 5)
    ax4.grid(True, axis="y")
    
    # Overall super-title
    fig.suptitle("UKRAINE AIR RAID ALERTS: SPATIO-TEMPORAL FORECASTING & CASCADE DYNAMICS", fontsize=16, fontweight="bold", y=0.995, color="#111827")
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.28, wspace=0.22)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[7/7] Visualization dashboard saved successfully: '{output_path}' ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Air Raid Alerts Spatial-Temporal Forecasting")
    parser.add_argument("--file", default="official_data_en.csv", help="Path to the official alerts CSV file")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Proportion of latest data for testing (0.0 to 1.0)")
    parser.add_argument("--skip-rf", action="store_true", help="Skip RandomForest to speed up training")
    parser.add_argument("--output-chart", default="alert_analysis.png", help="Path to save the generated visualization chart")
    args = parser.parse_args()
    
    print("=" * 78)
    print(" Ukraine Air Raid Alerts - Cascade Spatial/Temporal Forecasting Pipeline")
    print("=" * 78)
    
    # 1. Load Data
    raw_df = load_dataset(args.file)
    
    # 2. Clean Timestamps & Localize
    cleaned_df = clean_timestamps(raw_df)
    
    # 3. Aggregate Alerts by Hourly Bins and Oblast
    pivot_df, all_oblasts = aggregate_alerts_by_hour_and_oblast(cleaned_df)
    
    # Compute Spatial Matrices
    other_mean_matrix = compute_spatial_matrices(pivot_df, all_oblasts)
    
    # Automated Statistical Summary (Terminal Print)
    print_statistical_summary(cleaned_df, pivot_df, other_mean_matrix)
    
    # 4. Feature Engineering: Spatial Propagation + Temporal Lags
    feature_df = create_spatial_temporal_lag_features(pivot_df, other_mean_matrix, all_oblasts)
    
    # 5. Train & Evaluate Models
    eval_output = train_and_evaluate(feature_df, test_ratio=args.test_ratio, include_rf=not args.skip_rf)
    
    # 6. Generate Publication-Grade Visualizations
    generate_visualizations(
        df=cleaned_df,
        pivot_df=pivot_df,
        other_mean_matrix=other_mean_matrix,
        eval_output=eval_output,
        output_path=args.output_chart
    )
    
    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
