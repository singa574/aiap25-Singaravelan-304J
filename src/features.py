from __future__ import annotations

import numpy as np
import pandas as pd


LEAKAGE_OR_IDENTIFIER_COLUMNS = {
    "booking_id",              # identifier only
    "client_email",            # PII and near-unique identifier
    "is_canceled",             # target source
    "target",                  # label
    "reservation_status",      # outcome
    "reservation_status_date", # outcome timestamp
    "assigned_room_type",      # assigned at check-in
    "booking_changes",         # accumulated after booking creation
    "arrival_date",            # used for splitting; components are predictors
    "arrival_date_year",       # split key; a single train-year cannot teach extrapolation
    "arrival_date_month",      # replaced by a numeric month feature
}


def build_feature_frame(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create booking-time features, labels, and row metadata."""
    frame = clean.copy()
    frame["arrival_month"] = frame["arrival_date"].dt.month
    frame["arrival_day_of_week"] = frame["arrival_date"].dt.dayofweek
    frame["is_weekend_arrival"] = frame["arrival_day_of_week"].isin([5, 6]).astype("int8")
    frame["total_nights"] = (
        frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]
    )
    frame["total_guests"] = frame["adults"] + frame["children"] + frame["babies"]
    frame["has_children"] = (frame["children"].fillna(0) + frame["babies"].fillna(0)).gt(0).astype("int8")
    frame["adr_log1p"] = np.log1p(frame["adr"].clip(lower=0))

    metadata = frame[["booking_id", "hotel", "arrival_date", "reservation_status"]].copy()
    target = frame["target"].astype("int8")
    features = frame.drop(columns=sorted(LEAKAGE_OR_IDENTIFIER_COLUMNS & set(frame.columns)))
    # scikit-learn imputers expect NumPy NaN rather than pandas' scalar pd.NA
    # in mixed object arrays.
    for column in features.select_dtypes(include=["string", "object"]).columns:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    return features, target, metadata
