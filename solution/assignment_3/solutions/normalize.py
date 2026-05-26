"""Normalization helpers for Assignment 3 bus mapping.

The goal is conservative standardization: make names comparable without
destroying numeric identifiers that can distinguish separate buses.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


SEPARATOR_RE = re.compile(r"[-_./]+")
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")
KV_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*kv\b")


ABBREVIATIONS = {
    "substation": "sub",
    "station": "sta",
    "switch": "sw",
    "switching": "sw",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}


def normalize_bus_name(value: object) -> str:
    """Return a conservative normalized bus name."""
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = SEPARATOR_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def name_tokens(value: object) -> set[str]:
    """Tokenize a normalized bus name while preserving numeric tokens."""
    norm = normalize_bus_name(value)
    return {tok for tok in norm.split() if tok}


def numeric_tokens(value: object) -> set[str]:
    """Extract numeric tokens from a bus name."""
    return set(re.findall(r"\d+", normalize_bus_name(value)))


def voltage_band(kv: object) -> str:
    """Coarse voltage band used for compatibility checks."""
    try:
        value = float(kv)
    except (TypeError, ValueError):
        return "unknown"
    if pd.isna(value):
        return "unknown"
    if value >= 300:
        return "ehv"
    if value >= 100:
        return "hv"
    return "lv"


def coalesce_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first matching column, case-insensitively."""
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def standardize_bus_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Create a source-agnostic bus table with required normalized columns."""
    name_col = coalesce_column(df, ["NAME", "BUS_NAME", "bus_name"])
    kv_col = coalesce_column(df, ["BASKV", "KV", "VOLTAGE", "voltage_kv"])
    lat_col = coalesce_column(df, ["LAT", "LATITUDE", "latitude"])
    lon_col = coalesce_column(df, ["LON", "LONGITUDE", "longitude"])
    id_col = coalesce_column(df, ["BUS_ID", "PANO_ID", "ID", "MDB_ID"])

    if name_col is None:
        raise ValueError(f"{source} bus table does not contain a bus name column.")

    out = pd.DataFrame(index=df.index.copy())
    out[f"{source}_bus_id"] = (
        df[id_col].astype(str) + "_" + df.index.astype(str)
        if id_col is not None
        else df.index.astype(str)
    )
    out["raw_bus_name"] = df[name_col].astype(str)
    out["normalized_bus_name"] = df[name_col].map(normalize_bus_name)
    out["voltage_kv"] = pd.to_numeric(df[kv_col], errors="coerce") if kv_col else pd.NA
    out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.NA
    out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else pd.NA
    out["bus_source"] = source
    out["voltage_band"] = out["voltage_kv"].map(voltage_band)
    out["name_tokens"] = out["normalized_bus_name"].map(lambda x: " ".join(sorted(name_tokens(x))))
    out["numeric_tokens"] = out["normalized_bus_name"].map(lambda x: " ".join(sorted(numeric_tokens(x))))
    return out


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard overlap with a sensible empty-set fallback."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
