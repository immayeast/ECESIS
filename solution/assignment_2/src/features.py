from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out["hour"] = pd.to_numeric(out["he"], errors="coerce").astype(int) - 1
        out["timestamp"] = out["date"] + pd.to_timedelta(out["hour"], unit="h")
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["day_of_year"] = out["date"].dt.dayofyear
    out["year"] = out["date"].dt.year
    out["hour"] = pd.to_numeric(out["he"], errors="coerce").astype(int) - 1
    out = add_cyclical_time_features(out)
    return out


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic time fields with sine/cosine pairs."""
    out = df.copy()
    hour = pd.to_numeric(out["hour"], errors="coerce")
    dow = pd.to_numeric(out["day_of_week"], errors="coerce")
    doy = pd.to_numeric(out["day_of_year"], errors="coerce")
    month = pd.to_numeric(out["month"], errors="coerce")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 366)
    out["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 366)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "pd",
    lags: tuple[int, ...] = (24, 168),
    rolling_windows: tuple[int, ...] = (168, 672),
) -> pd.DataFrame:
    """Add shifted lag and rolling features for hourly panel data."""
    out = df.sort_values(group_cols + ["timestamp"]).copy()
    grouped = out.groupby(group_cols, sort=False)[target_col]
    for lag in lags:
        out[f"lag_{lag}h_{target_col}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    for window in rolling_windows:
        min_periods = max(24, min(window // 4, window))
        out[f"rolling_{window // 24}d_mean_{target_col}"] = shifted.groupby(
            [out[col] for col in group_cols], sort=False
        ).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
        out[f"rolling_{window // 24}d_std_{target_col}"] = shifted.groupby(
            [out[col] for col in group_cols], sort=False
        ).transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    return out


def add_expanding_historical_average(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "pd",
    output_col: str = "historical_avg_pd",
) -> pd.DataFrame:
    """Historical average using only previous rows inside each group."""
    out = df.sort_values(group_cols + ["timestamp"]).copy()
    grouped = out.groupby(group_cols, sort=False)[target_col]
    cumsum = grouped.cumsum() - out[target_col]
    count = grouped.cumcount()
    out[output_col] = cumsum / count.replace(0, np.nan)
    return out


def add_zone_context_from_bus_history(bus_df: pd.DataFrame) -> pd.DataFrame:
    out = bus_df.copy()
    zone_total = (
        out.groupby(["zone_name", "timestamp"], as_index=False)["pd"]
        .sum()
        .rename(columns={"pd": "zone_total_pd_actual"})
    )
    out = out.merge(zone_total, on=["zone_name", "timestamp"], how="left")
    out = out.sort_values(["zone_name", "timestamp", "bus_unique_id"])
    out["zone_total_pd_lag_24h"] = out.groupby("zone_name")["zone_total_pd_actual"].shift(24)
    out["zone_total_pd_lag_168h"] = out.groupby("zone_name")["zone_total_pd_actual"].shift(168)
    return out


def compute_historical_bus_share(
    train_bus: pd.DataFrame,
    by: list[str] | None = None,
) -> pd.DataFrame:
    by = by or ["zone_name", "bus_unique_id", "he", "day_of_week"]
    train = add_calendar_features(train_bus)
    zone_total = train.groupby(["zone_name", "timestamp"], as_index=False)["pd"].sum().rename(columns={"pd": "zone_pd"})
    train = train.merge(zone_total, on=["zone_name", "timestamp"], how="left")
    train["bus_share"] = train["pd"] / train["zone_pd"].replace(0, np.nan)
    share = train.groupby(by, as_index=False)["bus_share"].mean()
    zone_share_sum = share.groupby(["zone_name", "he", "day_of_week"])["bus_share"].transform("sum")
    share["bus_share"] = share["bus_share"] / zone_share_sum.replace(0, np.nan)
    return share


def attach_historical_bus_share(target_bus: pd.DataFrame, share: pd.DataFrame) -> pd.DataFrame:
    target = add_calendar_features(target_bus)
    key = ["zone_name", "bus_unique_id", "he", "day_of_week"]
    out = target.merge(share, on=key, how="left")
    fallback = share.groupby(["zone_name", "bus_unique_id"], as_index=False)["bus_share"].mean()
    out = out.merge(fallback, on=["zone_name", "bus_unique_id"], how="left", suffixes=("", "_fallback"))
    out["historical_bus_share"] = out["bus_share"].fillna(out["bus_share_fallback"])
    return out.drop(columns=[col for col in ["bus_share", "bus_share_fallback"] if col in out.columns])


def build_bus_feature_frame(bus: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(bus)
    out = add_lag_rolling_features(out, ["bus_unique_id"], lags=(24, 168), rolling_windows=(168, 672))
    out = add_expanding_historical_average(
        out,
        ["bus_unique_id", "he", "day_of_week"],
        target_col="pd",
        output_col="historical_avg_bus_he_dow_pd",
    )
    out = add_zone_context_from_bus_history(out)
    return out


def build_zone_feature_frame(zone: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(zone)
    out = add_lag_rolling_features(out, ["zone_name"], lags=(24, 168), rolling_windows=(168, 672))
    out = add_expanding_historical_average(
        out,
        ["zone_name", "he", "day_of_week"],
        target_col="pd",
        output_col="historical_avg_zone_he_dow_pd",
    )
    return out


def fit_group_mean(train: pd.DataFrame, group_cols: list[str], target_col: str, output_col: str) -> pd.DataFrame:
    mapping = train.groupby(group_cols, as_index=False)[target_col].mean()
    return mapping.rename(columns={target_col: output_col})


def apply_forecast_boundary(
    df: pd.DataFrame,
    forecast_created_at: str | pd.Timestamp,
    lag_hours: tuple[int, ...] = (24, 168),
    rolling_prefixes: tuple[str, ...] = ("rolling_",),
    target_col: str = "pd",
) -> pd.DataFrame:
    """Mask validation features whose source values would occur after forecast_created_at."""
    out = df.copy()
    boundary = pd.Timestamp(forecast_created_at)
    for lag in lag_hours:
        col = f"lag_{lag}h_{target_col}"
        if col in out.columns:
            source_time = out["timestamp"] - pd.to_timedelta(lag, unit="h")
            out.loc[source_time > boundary, col] = np.nan
    for col in out.columns:
        if any(col.startswith(prefix) for prefix in rolling_prefixes):
            out.loc[out["timestamp"] > boundary, col] = np.nan
    return out


def split_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str = "pd"):
    clean = df.dropna(subset=feature_cols + [target_col]).copy()
    return clean[feature_cols], clean[target_col], clean
