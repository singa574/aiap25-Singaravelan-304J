from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


TRUE_VALUES = {"1", "true", "yes"}
FALSE_VALUES = {"0", "false", "no"}
MISSING_TOKENS = {"", "-", "null", "none", "nan", "unknown", "undefined"}


def load_bookings(db_path: str | Path, table: str = "bookings") -> pd.DataFrame:
    """Read a complete SQLite table without modifying the source database."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Database not found at {path}. Place hotel.db in data/ or pass --db-path."
        )
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"Unsafe SQLite table name: {table!r}")

    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        available = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if table not in available:
            raise ValueError(f"Table {table!r} not found. Available tables: {sorted(available)}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.mask(cleaned.str.lower().isin(MISSING_TOKENS))


def _normalise_category(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    return cleaned.str.replace(r"\s*-\s*", "-", regex=True).str.lower()


def _parse_binary(series: pd.Series, name: str) -> pd.Series:
    cleaned = _clean_text(series).str.lower()
    parsed = pd.Series(pd.NA, index=series.index, dtype="Int8")
    parsed.loc[cleaned.isin(TRUE_VALUES)] = 1
    parsed.loc[cleaned.isin(FALSE_VALUES)] = 0
    unknown = cleaned.notna() & parsed.isna()
    if unknown.any():
        values = sorted(cleaned.loc[unknown].unique().tolist())
        raise ValueError(f"Unrecognised values in {name}: {values}")
    return parsed


def clean_bookings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply deterministic, target-independent cleaning and integrity checks."""
    frame = raw.copy()
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns="Unnamed: 0")

    required = {
        "booking_id", "client_email", "is_canceled", "reservation_status",
        "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required booking columns: {sorted(missing)}")

    frame["booking_id"] = _clean_text(frame["booking_id"])
    rows_before = len(frame)
    duplicate_ids = int(frame["booking_id"].duplicated(keep="first").sum())
    frame = frame.drop_duplicates(subset="booking_id", keep="first").copy()

    frame["target"] = _parse_binary(frame["is_canceled"], "is_canceled")
    status = _clean_text(frame["reservation_status"]).str.lower()
    status_target = status.map({"check-out": 0, "canceled": 1, "no-show": 1}).astype("Int8")
    mismatch = frame["target"].notna() & status_target.notna() & frame["target"].ne(status_target)
    if mismatch.any():
        raise ValueError(f"Target/status mismatch in {int(mismatch.sum())} rows")
    if frame["target"].isna().any():
        raise ValueError("Target contains missing values after normalisation")

    numeric_columns = [
        "lead_time", "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "days_in_waiting_list", "adr",
        "required_car_parking_spaces", "total_of_special_requests",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(_clean_text(frame[column]), errors="coerce")

    # A negative interval between booking and arrival is impossible at booking time.
    negative_lead_time = frame["lead_time"].lt(0).fillna(False)
    frame["lead_time_invalid"] = negative_lead_time.astype("int8")
    frame.loc[frame["lead_time"].lt(0), "lead_time"] = np.nan

    month = _clean_text(frame["arrival_date_month"]).str[:3].str.title()
    date_parts = (
        frame["arrival_date_year"].astype("Int64").astype("string")
        + "-" + month + "-"
        + frame["arrival_date_day_of_month"].astype("Int64").astype("string")
    )
    frame["arrival_date"] = pd.to_datetime(date_parts, format="%Y-%b-%d", errors="coerce")
    if frame["arrival_date"].isna().any():
        raise ValueError(f"Could not construct arrival date for {int(frame['arrival_date'].isna().sum())} rows")

    categorical_columns = [
        "hotel", "meal", "country", "market_segment", "distribution_channel",
        "reserved_room_type", "deposit_type", "agent", "company", "customer_type",
        "promo_code",
    ]
    for column in categorical_columns:
        frame[column] = _normalise_category(frame[column])

    # Canonicalise frequent spelling variants observed during EDA.
    aliases = {
        "meal": {"bb": "bb", "hb": "hb", "fb": "fb", "sc": "sc"},
        "market_segment": {"onlin ta": "online ta", "online-ta": "online ta", "offline ta/t0": "offline ta/to"},
        "deposit_type": {"no depost": "no deposit", "no-deposit": "no deposit", "non-refund": "non refund"},
        "customer_type": {"transiant": "transient"},
    }
    for column, mapping in aliases.items():
        frame[column] = frame[column].replace(mapping)

    audit = {
        "raw_rows": rows_before,
        "duplicate_booking_ids_removed": duplicate_ids,
        "clean_rows": len(frame),
    }
    return frame.reset_index(drop=True), audit
