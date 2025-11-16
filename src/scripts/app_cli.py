import argparse
import sys
from pathlib import Path

# Add proper path handling
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import run_pipeline
from src.config import AppConfig, load_config
from src.utils.logger import setup_logger, get_logger


def main():
    """Main CLI entry point with enhanced error handling and logging."""
    parser = argparse.ArgumentParser(
        description="Multi-modal anomaly detection prototype for oil rig operations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where results (CSVs and summary) will be stored.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Approximate proportion of anomalies in the data for IsolationForest.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for sensor data simulation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-03-31",
        help="End date for sensor data simulation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file name.",
    )
    parser.add_argument(
        "--enable-genai",
        action="store_true",
        help="Enable GenAI features using Ollama (requires Ollama to be running).",
    )
    parser.add_argument(
        "--genai-model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (e.g., llama3.2:3b, llama2, mistral).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(
        name="oil_rig_cli",
        level=log_level,
        log_file=args.log_file
    )
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            config = AppConfig()
        
        # Override config with CLI arguments
        config.data.start_date = args.start_date
        config.data.end_date = args.end_date
        config.model.contamination = args.contamination
        config.output.output_dir = args.output_dir
        
        # Configure GenAI settings
        if args.enable_genai:
            config.genai.enabled = True
            config.genai.model = args.genai_model
            config.genai.base_url = args.ollama_url
            logger.info(f"GenAI enabled: {args.genai_model} @ {args.ollama_url}")
        
        logger.info("="*60)
        logger.info("Starting Oil Rig Anomaly Detection Pipeline")
        logger.info("="*60)
        logger.info(f"Date Range: {args.start_date} to {args.end_date}")
        logger.info(f"Contamination Rate: {args.contamination:.2%}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"GenAI Mode: {'ENABLED' if config.genai.enabled else 'DISABLED'}")
        if config.genai.enabled:
            logger.info(f"  Model: {config.genai.model}")
        logger.info("="*60)
        
        # Prepare sensor parameters
        sensor_params = {
            "start": args.start_date,
            "end": args.end_date,
            "freq": "1H",
        }
        
        # Run pipeline
        logger.info("Executing pipeline...")
        sensor_df, logs_df, anomalies_df, correlated_df, report_text = run_pipeline(
            output_dir=args.output_dir,
            sensor_params=sensor_params,
            contamination=args.contamination,
            config=config,
        )
        
        # Calculate statistics
        total_points = len(sensor_df)
        total_anomalies = anomalies_df['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(anomalies_df)) * 100
        n_logs = len(logs_df)
        n_equipment = sensor_df['equipment_id'].nunique()
        
        logger.info("="*60)
        logger.info("Pipeline Completed Successfully!")
        logger.info("="*60)
        logger.info(f"Total sensor data points: {total_points:,}")
        logger.info(f"Equipment monitored: {n_equipment}")
        logger.info(f"Anomalies detected: {total_anomalies:,} ({anomaly_rate:.2f}%)")
        logger.info(f"Operator logs generated: {n_logs:,}")
        logger.info(f"Correlated log entries: {len(correlated_df):,}")
        logger.info("="*60)
        
        # Display summary preview
        print("\n" + "="*60)
        print("SUMMARY REPORT (Preview - First 20 Lines)")
        print("="*60)
        print("\n".join(report_text.splitlines()[:20]))
        print("\n" + "="*60)
        print(f"📁 Full results saved to: {Path(args.output_dir).absolute()}")
        print("="*60)
        print("\nFiles generated:")
        print(f"  • sensor_data.csv      - {total_points:,} sensor readings")
        print(f"  • operator_logs.csv    - {n_logs:,} log entries")
        print(f"  • anomalies.csv        - {total_anomalies:,} detected anomalies")
        print(f"  • correlated_logs.csv  - {len(correlated_df):,} log correlations")
        print(f"  • summary.txt          - Complete analysis report")
        print("="*60 + "\n")
        
        logger.info("All outputs written successfully")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("Check logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
