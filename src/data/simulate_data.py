import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime

RNG = np.random.default_rng(42)

EQUIPMENT_IDS = ["pump_1", "pump_2", "compressor_1", "compressor_2"]
SENSOR_TYPES = ["pressure", "temperature", "vibration"]


def _generate_base_signal(n: int, sensor: str) -> np.ndarray:
    """Create a simple base signal for a sensor type."""
    t = np.linspace(0, 2 * np.pi, n)
    if sensor == "pressure":
        base = 50 + 5 * np.sin(2 * t)
    elif sensor == "temperature":
        base = 80 + 3 * np.sin(t)  # mild oscillation
    elif sensor == "vibration":
        base = 10 + 2 * np.sin(3 * t)
    else:
        base = np.zeros(n)
    return base


def _inject_anomalies(values: np.ndarray, n_gradual: int = 3, n_sudden: int = 5) -> np.ndarray:
    """Inject gradual and sudden anomalies into the signal."""
    n = len(values)
    values = values.copy()

    # Gradual anomalies: drifts over a window
    for _ in range(n_gradual):
        start = RNG.integers(0, max(1, n - 24))
        length = RNG.integers(6, 24)  # ~6-24 time steps
        drift = RNG.uniform(5, 15) * (1 if RNG.random() > 0.5 else -1)
        end = min(n, start + length)
        ramp = np.linspace(0, drift, end - start)
        values[start:end] += ramp

    # Sudden anomalies: spikes or drops at a single timestep
    for _ in range(n_sudden):
        idx = RNG.integers(0, n)
        jump = RNG.uniform(10, 25) * (1 if RNG.random() > 0.5 else -1)
        values[idx] += jump

    return values


def generate_sensor_data(
    start: str = "2024-01-01",
    end: str = "2024-03-31",
    freq: str = "1H",
    equipment_ids: List[str] = None,
    sensor_types: List[str] = None,
) -> pd.DataFrame:
    """Generate synthetic time series sensor data.

    Returns a DataFrame with columns:
    [timestamp, equipment_id, sensor_type, value]
    """
    if equipment_ids is None:
        equipment_ids = EQUIPMENT_IDS
    if sensor_types is None:
        sensor_types = SENSOR_TYPES

    timestamps = pd.date_range(start=start, end=end, freq=freq)
    n = len(timestamps)

    records = []
    for eq in equipment_ids:
        for sensor in sensor_types:
            base = _generate_base_signal(n, sensor)
            noise = RNG.normal(0, 1.0, size=n)
            values = base + noise
            values = _inject_anomalies(values)

            # Inject some missing values randomly
            mask_missing = RNG.random(n) < 0.02  # 2% missing
            values = values.astype(float)
            values[mask_missing] = np.nan

            for ts, val in zip(timestamps, values):
                records.append(
                    {
                        "timestamp": ts,
                        "equipment_id": eq,
                        "sensor_type": sensor,
                        "value": val,
                    }
                )

    df = pd.DataFrame.from_records(records)
    return df


OBS_TEMPLATES = [
    "Noticed spike in {sensor} on {equipment}.",
    "Slight abnormal behaviour: {sensor} readings off on {equipment}.",
    "Operator heard unusual noise from {equipment}.",
    "Routine check on {equipment}; all looks normal.",
    "Possible leak issue near {equipment}.",
    "Sudden jump in {sensor} for {equipment}, needs follow-up.",
    "Observed gradual drift in {sensor} values for {equipment}.",
]


def generate_operator_logs(
    sensor_df: pd.DataFrame,
    max_logs_per_day: Tuple[int, int] = (1, 5),
) -> pd.DataFrame:
    """Generate synthetic operator logs based on sensor data statistics.

    Creates more logs around days with higher anomaly-like variation.
    """
    # Use per-day per-equipment volatility to influence log density
    df = sensor_df.copy()
    df["date"] = df["timestamp"].dt.date

    volatility = (
        df.groupby(["date", "equipment_id", "sensor_type"])["value"]
        .std()
        .reset_index(name="volatility")
    )

    # Normalise volatility to get sampling weights
    volatility["volatility"].fillna(0.0, inplace=True)
    max_vol = volatility["volatility"].max() or 1.0
    volatility["weight"] = 0.2 + 0.8 * (volatility["volatility"] / max_vol)

    logs = []
    for (date, eq), group in volatility.groupby(["date", "equipment_id"]):
        day_weight = group["weight"].mean()
        # Decide how many logs for this day/equipment
        n_logs = RNG.integers(max_logs_per_day[0], max_logs_per_day[1] + 1)
        n_logs = int(np.round(n_logs * day_weight))

        if n_logs <= 0:
            continue

        # Get timestamps for that day/equipment to sample from
        mask = (df["equipment_id"] == eq) & (df["date"] == date)
        day_ts = df.loc[mask, "timestamp"].values
        if len(day_ts) == 0:
            continue

        for _ in range(n_logs):
            ts = RNG.choice(day_ts)
            sensor = RNG.choice(SENSOR_TYPES)
            template = RNG.choice(OBS_TEMPLATES)
            text = template.format(sensor=sensor, equipment=eq)
            logs.append(
                {
                    "timestamp": pd.to_datetime(ts),
                    "equipment_id": eq,
                    "log_text": text,
                }
            )

    logs_df = pd.DataFrame(logs)
    logs_df.sort_values("timestamp", inplace=True)
    logs_df.reset_index(drop=True, inplace=True)
    return logs_df


if __name__ == "__main__":
    sensor_df = generate_sensor_data()
    logs_df = generate_operator_logs(sensor_df)
    print(sensor_df.head())
    print(logs_df.head())
