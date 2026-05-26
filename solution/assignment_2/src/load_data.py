from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


BUS_COLUMNS = ["id", "bus_unique_id", "bus_type", "base_kv", "zone_name", "pd", "pg", "date", "he"]
ZONE_COLUMNS = ["zone_name", "pd", "pg", "load_bus_count", "gen_bus_count", "date", "he"]


def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root containing README.md and data/.")


def assignment2_data_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "data" / "Assignment 2 - forecast"


def assignment2_output_dir(repo_root: Path | None = None, name: str = "outputs") -> Path:
    root = repo_root or find_repo_root()
    out = root / "solution" / "assignment_2" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_year_files(kind: str, data_dir: Path | None = None) -> dict[int, Path]:
    if kind not in {"bus", "zone"}:
        raise ValueError("kind must be 'bus' or 'zone'")
    base = data_dir or assignment2_data_dir()
    files = {}
    for path in sorted(base.glob(f"{kind}_load_*.parquet")):
        year = int(path.stem.split("_")[-1])
        files[year] = path
    if not files:
        raise FileNotFoundError(f"No {kind}_load_*.parquet files found under {base}")
    return files


def parquet_inventory(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or assignment2_data_dir()
    rows = []
    for path in sorted(base.glob("*_load_*.parquet")):
        pf = pq.ParquetFile(path)
        rows.append(
            {
                "file": path.name,
                "kind": "bus" if path.name.startswith("bus_") else "zone",
                "rows": pf.metadata.num_rows,
                "row_groups": pf.num_row_groups,
                "columns": ",".join(pf.schema_arrow.names),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            }
        )
    return pd.DataFrame(rows)


def _date_filter(start_date: str | None = None, end_date: str | None = None):
    expr = None
    if start_date is not None:
        expr = ds.field("date") >= pd.Timestamp(start_date).strftime("%Y-%m-%d")
    if end_date is not None:
        upper = ds.field("date") <= pd.Timestamp(end_date).strftime("%Y-%m-%d")
        expr = upper if expr is None else expr & upper
    return expr


def read_years(
    kind: str,
    years: Iterable[int],
    columns: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    bus_ids: Iterable[str] | None = None,
    zones: Iterable[str] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Read selected Assignment 2 parquet files with column and date pruning."""
    files = list_year_files(kind, data_dir)
    frames = []
    for year in years:
        if year not in files:
            continue
        dataset = ds.dataset(files[year], format="parquet")
        filter_expr = _date_filter(start_date, end_date)
        if bus_ids is not None:
            bus_expr = ds.field("bus_unique_id").isin(list(bus_ids))
            filter_expr = bus_expr if filter_expr is None else filter_expr & bus_expr
        if zones is not None:
            zone_expr = ds.field("zone_name").isin(list(zones))
            filter_expr = zone_expr if filter_expr is None else filter_expr & zone_expr
        table = dataset.to_table(columns=columns, filter=filter_expr)
        frame = table.to_pandas()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def add_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    out["date"] = pd.to_datetime(out["date"])
    out["he"] = pd.to_numeric(out["he"], errors="coerce").astype("Int64")
    out["hour"] = out["he"].astype(int) - 1
    out["timestamp"] = out["date"] + pd.to_timedelta(out["hour"], unit="h")
    return out


def read_bus(
    years: Iterable[int],
    columns: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    bus_ids: Iterable[str] | None = None,
    zones: Iterable[str] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    cols = columns or BUS_COLUMNS
    df = read_years("bus", years, cols, start_date, end_date, bus_ids, zones, data_dir)
    return add_timestamp_columns(df) if not df.empty else df


def read_zone(
    years: Iterable[int],
    columns: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    zones: Iterable[str] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    cols = columns or ZONE_COLUMNS
    df = read_years("zone", years, cols, start_date, end_date, None, zones, data_dir)
    return add_timestamp_columns(df) if not df.empty else df


def first_buses_by_volume(
    years: Iterable[int],
    n_buses: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: Path | None = None,
) -> list[str]:
    """Choose deterministic high-volume buses for prototype runs without random sampling."""
    df = read_bus(
        years,
        columns=["bus_unique_id", "pd", "date", "he"],
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )
    if df.empty:
        return []
    volume = df.groupby("bus_unique_id")["pd"].sum(min_count=1).sort_values(ascending=False)
    return volume.head(n_buses).index.astype(str).tolist()


def first_buses_from_first_batch(
    year: int = 2022,
    n_buses: int = 200,
    data_dir: Path | None = None,
) -> list[str]:
    """Deterministic lightweight bus subset for prototype runs."""
    files = list_year_files("bus", data_dir)
    pf = pq.ParquetFile(files[year])
    batch = next(pf.iter_batches(batch_size=200_000, columns=["bus_unique_id", "pd"]))
    df = batch.to_pandas()
    volume = df.groupby("bus_unique_id")["pd"].sum(min_count=1).sort_values(ascending=False)
    return volume.head(n_buses).index.astype(str).tolist()


def reconcile_zone_to_bus(bus: pd.DataFrame, zone: pd.DataFrame) -> pd.DataFrame:
    """Compare zone pd to the sum of bus pd by zone and timestamp."""
    bus_sum = (
        bus.groupby(["zone_name", "date", "he", "timestamp"], as_index=False)["pd"]
        .sum()
        .rename(columns={"pd": "bus_sum_pd"})
    )
    zone_small = zone[["zone_name", "date", "he", "timestamp", "pd"]].rename(columns={"pd": "zone_pd"})
    out = zone_small.merge(bus_sum, on=["zone_name", "date", "he", "timestamp"], how="left")
    out["abs_diff_pd"] = (out["zone_pd"] - out["bus_sum_pd"]).abs()
    out["pct_diff"] = out["abs_diff_pd"] / out["zone_pd"].abs().replace(0, pd.NA)
    return out
