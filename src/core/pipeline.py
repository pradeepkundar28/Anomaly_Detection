"""
Enhanced pipeline with configurable GenAI support.
"""
import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from src.data import generate_sensor_data, generate_operator_logs
from src.core.detection import detect_anomalies
from src.core.processing import build_log_embeddings, correlate_logs_with_anomalies
from src.core.insights import generate_insight_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    output_dir: str = "output",
    sensor_params: Dict[str, Any] = None,
    contamination: float = 0.03,
    config: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run the end-to-end anomaly detection and insight generation pipeline.
    
    Args:
        output_dir: Directory to save outputs
        sensor_params: Parameters for sensor data generation
        contamination: Anomaly contamination rate
        config: AppConfig object (optional, for GenAI features)
        
    Returns:
        sensor_df, logs_df, anomalies_df, correlated_df, report_text
    """
    if sensor_params is None:
        sensor_params = {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "freq": "1H",
        }

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Starting Anomaly Detection Pipeline")
    if config and hasattr(config, 'genai') and config.genai.enabled:
        logger.info(f"GenAI Mode: ENABLED (using {config.genai.model})")
    else:
        logger.info("GenAI Mode: DISABLED (using traditional rule-based)")
    logger.info("="*60)

    # 1. Simulate data
    logger.info("[1/5] Generating sensor data...")
    sensor_df = generate_sensor_data(**sensor_params)
    logger.info(f"  ✓ Generated {len(sensor_df):,} sensor readings")
    
    logger.info("[2/5] Generating operator logs...")
    logs_df = generate_operator_logs(sensor_df)
    logger.info(f"  ✓ Generated {len(logs_df):,} operator logs")

    # 2. Detect anomalies
    logger.info("[3/5] Detecting anomalies...")
    anomalies_df, model = detect_anomalies(sensor_df, contamination=contamination)
    n_anomalies = anomalies_df['is_anomaly'].sum()
    logger.info(f"  ✓ Detected {n_anomalies:,} anomalies")

    # 3. Text embeddings + correlation
    logger.info("[4/5] Correlating logs with anomalies...")
    if logs_df.empty:
        correlated_df = pd.DataFrame()
        report_text = generate_insight_report(anomalies_df, correlated_df, logs_df, config)
    else:
        vectorizer, log_embeddings = build_log_embeddings(logs_df)
        correlated_df = correlate_logs_with_anomalies(
            anomalies_df=anomalies_df,
            logs_df=logs_df,
            vectorizer=vectorizer,
            log_embeddings=log_embeddings,
        )
        logger.info(f"  ✓ Found {len(correlated_df):,} correlations")

        # 4. Insight generation
        logger.info("[5/5] Generating insights...")
        report_text = generate_insight_report(anomalies_df, correlated_df, logs_df, config)
        logger.info("  ✓ Generated insight report")

    # 5. Save artifacts
    sensor_path = os.path.join(output_dir, "sensor_data.csv")
    logs_path = os.path.join(output_dir, "operator_logs.csv")
    anomalies_path = os.path.join(output_dir, "anomalies.csv")
    correlated_path = os.path.join(output_dir, "correlated_logs.csv")
    summary_path = os.path.join(output_dir, "summary.txt")

    sensor_df.to_csv(sensor_path, index=False)
    logs_df.to_csv(logs_path, index=False)
    anomalies_df.to_csv(anomalies_path, index=False)
    correlated_df.to_csv(correlated_path, index=False)
    with open(summary_path, "w") as f:
        f.write(report_text)
    
    logger.info(f"  ✓ Results saved to {output_dir}/")
    logger.info("="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)

    return sensor_df, logs_df, anomalies_df, correlated_df, report_text
