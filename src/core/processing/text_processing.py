from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_log_embeddings(logs_df: pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Build TF-IDF embeddings for log_text column."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    embeddings = vectorizer.fit_transform(logs_df["log_text"].fillna(""))
    return vectorizer, embeddings


def _build_anomaly_query(row: pd.Series, sensor_columns) -> str:
    """Create a simple textual description for an anomaly row."""
    equipment_id = row["equipment_id"]
    # Identify sensor with largest absolute deviation from median-of-row
    sensors = {s: row[s] for s in sensor_columns}
    values = np.array(list(sensors.values()))
    median = np.median(values)
    deviations = {s: abs(v - median) for s, v in sensors.items()}
    main_sensor = max(deviations, key=deviations.get)

    query = (
        f"Anomaly detected in {equipment_id}, mainly affecting {main_sensor} sensor. "
        f"Abnormal readings across sensors: "
        + ", ".join(f"{s}={row[s]:.2f}" for s in sensor_columns)
    )
    return query


def correlate_logs_with_anomalies(
    anomalies_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    log_embeddings,
    time_window: pd.Timedelta = pd.Timedelta("2H"),
    top_k: int = 3,
    similarity_threshold: float = 0.1,
) -> pd.DataFrame:
    """For each anomaly, find top-k semantically/time-relevant logs.

    Returns a DataFrame where each row is an anomaly-log pair with similarity score.
    """
    # Ensure timestamp types
    anomalies = anomalies_df.copy()
    logs = logs_df.copy()
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])

    sensor_columns = [
        c for c in anomalies.columns
        if c not in ["timestamp", "equipment_id", "anomaly_score", "is_anomaly"]
    ]

    records = []

    for idx, row in anomalies[anomalies["is_anomaly"]].iterrows():
        ts = row["timestamp"]
        eq = row["equipment_id"]

        # Filter logs by equipment and time window
        mask = (logs["equipment_id"] == eq) & (
            (logs["timestamp"] >= ts - time_window)
            & (logs["timestamp"] <= ts + time_window)
        )
        candidate_logs = logs[mask]

        if candidate_logs.empty:
            continue

        query_text = _build_anomaly_query(row, sensor_columns)
        query_vec = vectorizer.transform([query_text])

        # subset embeddings
        candidate_indices = candidate_logs.index.to_list()
        log_vecs = log_embeddings[candidate_indices]

        sims = cosine_similarity(query_vec, log_vecs).flatten()
        top_idx = np.argsort(-sims)[:top_k]

        for rank in top_idx:
            sim = sims[rank]
            if sim < similarity_threshold:
                continue
            log_row = candidate_logs.iloc[rank]
            records.append(
                {
                    "anomaly_timestamp": ts,
                    "anomaly_equipment_id": eq,
                    "anomaly_score": row["anomaly_score"],
                    "log_timestamp": log_row["timestamp"],
                    "log_equipment_id": log_row["equipment_id"],
                    "log_text": log_row["log_text"],
                    "similarity": float(sim),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "anomaly_timestamp",
                "anomaly_equipment_id",
                "anomaly_score",
                "log_timestamp",
                "log_equipment_id",
                "log_text",
                "similarity",
            ]
        )

    result = pd.DataFrame.from_records(records)
    result.sort_values(["anomaly_timestamp", "similarity"], ascending=[True, False], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
