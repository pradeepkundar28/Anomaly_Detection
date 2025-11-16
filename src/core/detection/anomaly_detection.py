from typing import Tuple
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


def preprocess_sensor_data(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sensor data into wide format and handle missing values.

    Input columns: timestamp, equipment_id, sensor_type, value
    Output index: timestamp, equipment_id
    Columns: one per sensor_type
    """
    wide = sensor_df.pivot_table(
        index=["timestamp", "equipment_id"],
        columns="sensor_type",
        values="value",
    ).reset_index()

    # Simple missing value handling: forward fill within equipment, then backfill
    wide = wide.sort_values(["equipment_id", "timestamp"])
    sensor_cols = [c for c in wide.columns if c not in ["timestamp", "equipment_id"]]
    wide[sensor_cols] = (
        wide.groupby("equipment_id")[sensor_cols]
        .apply(lambda g: g.ffill().bfill())
        .reset_index(level=0, drop=True)
    )

    # If any remaining NaNs, fill with column medians
    for c in sensor_cols:
        median = wide[c].median()
        wide[c].fillna(median, inplace=True)

    return wide


def detect_anomalies(
    sensor_df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, IsolationForest]:
    """Run IsolationForest-based anomaly detection.

    Returns:
        anomalies_df: wide data with anomaly scores & flags
        model: trained IsolationForest instance
    """
    wide = preprocess_sensor_data(sensor_df)
    sensor_cols = [c for c in wide.columns if c not in ["timestamp", "equipment_id"]]

    X = wide[sensor_cols].values

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    # decision_function: higher is more normal, so we negate to get anomaly_score
    decision_scores = model.decision_function(X)
    anomaly_score = -decision_scores
    is_anomaly = model.predict(X) == -1

    wide["anomaly_score"] = anomaly_score
    wide["is_anomaly"] = is_anomaly

    return wide, model


def attach_anomaly_flags(sensor_df: pd.DataFrame, anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Join anomaly flags back onto the long-form sensor_df if needed."""
    merged = sensor_df.merge(
        anomalies_df[["timestamp", "equipment_id", "anomaly_score", "is_anomaly"]],
        on=["timestamp", "equipment_id"],
        how="left",
    )
    return merged
