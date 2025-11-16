"""
Enhanced insight generation with configurable GenAI support using Ollama.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _summarise_anomalies(anomalies_df: pd.DataFrame) -> str:
    total_points = len(anomalies_df)
    total_anomalies = anomalies_df["is_anomaly"].sum()
    rate = (total_anomalies / max(1, total_points)) * 100.0

    by_eq = (
        anomalies_df[anomalies_df["is_anomaly"]]
        .groupby("equipment_id")["is_anomaly"]
        .count()
        .sort_values(ascending=False)
    )

    lines = []
    lines.append(f"Total data points analysed: {total_points}")
    lines.append(f"Total anomalies detected: {total_anomalies} ({rate:.2f}% of points)\n")

    if not by_eq.empty:
        lines.append("Anomaly counts by equipment (descending):")
        for eq, cnt in by_eq.items():
            lines.append(f"  - {eq}: {cnt} anomalies"
                         )

    return "\n".join(lines)


def _hypothesise_root_causes(anomalies_df: pd.DataFrame) -> str:
    """Very simple rule-based root-cause hypotheses based on sensor involvement."""
    sensor_cols = [c for c in anomalies_df.columns if c not in ["timestamp", "equipment_id", "anomaly_score", "is_anomaly"]]

    anomalies = anomalies_df[anomalies_df["is_anomaly"]]
    if anomalies.empty or not sensor_cols:
        return "Not enough anomalies to form meaningful hypotheses yet."

    # Approximate sensor "importance" per anomaly by deviation from per-equipment median
    deviations = {s: 0.0 for s in sensor_cols}
    for _, row in anomalies.iterrows():
        vals = np.array([row[s] for s in sensor_cols])
        median = np.median(vals)
        for s in sensor_cols:
            deviations[s] += abs(row[s] - median)

    sorted_sensors = sorted(deviations.items(), key=lambda x: x[1], reverse=True)
    lines = []
    lines.append("Likely contributing factors (rule-based approximation):")


    for sensor, score in sorted_sensors:
        if sensor.lower().startswith("pressure") or sensor == "pressure":
            hint = "Possible causes: line blockage, valve issues, or pump overload."
        elif sensor.lower().startswith("temperature") or sensor == "temperature":
            hint = "Possible causes: cooling failure, lubrication issues, or overheating."
        elif sensor.lower().startswith("vibration") or sensor == "vibration":
            hint = "Possible causes: mechanical imbalance, misalignment, or bearing wear."
        else:
            hint = "Generic sensor anomaly; further inspection required."

        lines.append(f"  - {sensor}: high anomaly involvement. {hint}")

    return "\n".join(lines)


def generate_insight_report(
    anomalies_df: pd.DataFrame,
    correlated_logs_df: pd.DataFrame,
    logs_df: Optional[pd.DataFrame] = None,
    config: Optional[object] = None,
) -> str:
    """
    Generate a human-readable insight report with optional GenAI enhancement.
    
    Args:
        anomalies_df: DataFrame of detected anomalies
        correlated_logs_df: DataFrame of correlated logs
        logs_df: Optional full logs DataFrame (needed for GenAI features)
        config: AppConfig object with genai settings
        
    Returns:
        Formatted insight report text
    """
    use_genai = False
    genai_service = None
    
    # Check if GenAI is enabled and available
    if config and hasattr(config, 'genai') and config.genai.enabled:
        try:
            from src.core.genai.llm_service import OllamaLLMService, check_ollama_available
            
            if check_ollama_available(config.genai.base_url):
                genai_service = OllamaLLMService(
                    model=config.genai.model,
                    base_url=config.genai.base_url,
                    temperature=config.genai.temperature,
                    max_tokens=config.genai.max_tokens
                )
                use_genai = True
                logger.info("GenAI mode enabled using Ollama")
            else:
                logger.warning(f"Ollama not available at {config.genai.base_url}")
                if config.genai.fallback_to_traditional:
                    logger.info("Falling back to traditional rule-based insights")
                else:
                    return "Error: GenAI enabled but Ollama not available. Run 'ollama serve' first."
        except Exception as e:
            logger.error(f"Failed to initialize GenAI: {e}")
            if config.genai.fallback_to_traditional:
                logger.info("Falling back to traditional rule-based insights")
            else:
                return f"Error initializing GenAI: {str(e)}"
    
    # Build report header
    header = (
        "="*70 + "\n"+
        "OIL RIG ANOMALY DETECTION - INSIGHT REPORT\n"+
        "="*70 + "\n"+
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Mode: {'GenAI-Enhanced (Ollama)' if use_genai else 'Traditional (Rule-Based)'}\n"
        "\n"
    )
    
    # Generate summary statistics
    summary = _summarise_anomalies(anomalies_df)
    
    report_sections = [header, "SUMMARY STATISTICS\n" + "-"*70, summary, ""]
    
    # GenAI-Enhanced Analysis
    if use_genai and genai_service:
        report_sections.extend(_generate_genai_insights(
            genai_service, anomalies_df, correlated_logs_df, logs_df, config
        ))
    
    # Traditional rule-based analysis (always included as baseline)
    report_sections.extend([
        "\nTRADITIONAL ANALYSIS (RULE-BASED)\n" + "-"*70,
        _hypothesise_root_causes(anomalies_df),
        ""
    ])
    
    # Correlation insights
    if correlated_logs_df is None or correlated_logs_df.empty:
        corr_section = (
            "\nLOG CORRELATION\n" + "-"*70 + "\n"
            "No operator logs were strongly correlated with detected anomalies "
            "within the configured time and similarity windows.\n"
        )
    else:
        corr_lines = ["\nLOG CORRELATION\n" + "-"*70, "Sample anomaly ↔ log correlations (top 5):"]
        sample = correlated_logs_df.nlargest(5, 'similarity')
        for idx, row in sample.iterrows():
            corr_lines.append(
                f"\n{idx+1}. {row['anomaly_timestamp']} [{row['anomaly_equipment_id']}] "
                f"<-> log at {row['log_timestamp']}:\n"
                f"   '{row['log_text']}'\n"
                f"   (similarity={row['similarity']:.3f})"
            )
        corr_section = "\n".join(corr_lines)
    
    report_sections.append(corr_section)
    
    # Footer
    footer = (
        "\n" + "="*70 + "\n" +
        "END OF REPORT\n" +
        "="*70
    )
    report_sections.append(footer)
    
    return "\n".join(report_sections)


def _generate_genai_insights(
    genai_service,
    anomalies_df: pd.DataFrame,
    correlated_logs_df: pd.DataFrame,
    logs_df: Optional[pd.DataFrame],
    config: object
) -> list:
    """Generate GenAI-powered insights sections."""
    sections = []
    
    # Executive Summary
    if config.genai.enable_executive_summary and logs_df is not None:
        sections.append("\nEXECUTIVE SUMMARY (AI-GENERATED)\n" + "-"*70)
        try:
            logger.info("Generating executive summary with GenAI...")
            exec_summary = genai_service.generate_executive_summary(anomalies_df, logs_df)
            sections.append(exec_summary)
            sections.append("")
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            sections.append(f"[Executive summary generation failed: {str(e)}]\n")
    
    # Detailed Anomaly Analysis
    if config.genai.enable_anomaly_analysis:
        sections.append("\nTOP ANOMALIES - AI ANALYSIS\n" + "-"*70)
        
        # Analyze top 3 anomalies
        top_anomalies = anomalies_df[anomalies_df['is_anomaly']].nlargest(3, 'anomaly_score')
        
        for idx, (_, anomaly) in enumerate(top_anomalies.iterrows(), 1):
            sections.append(f"\n{idx}. {anomaly['equipment_id']} at {anomaly['timestamp']}")
            sections.append(f"   Anomaly Score: {anomaly['anomaly_score']:.3f}")
            sections.append("")
            
            try:
                # Get related logs
                related_logs = correlated_logs_df[
                    correlated_logs_df['anomaly_timestamp'] == anomaly['timestamp']
                ]['log_text'].tolist()
                
                logger.info(f"Analyzing anomaly {idx}/3 with GenAI...")
                analysis = genai_service.analyze_anomaly(anomaly.to_dict(), related_logs)
                sections.append(analysis)
                sections.append("")
            except Exception as e:
                logger.error(f"Anomaly analysis {idx} failed: {e}")
                sections.append(f"[Analysis failed: {str(e)}]\n")
    
    # Maintenance Predictions
    if config.genai.enable_maintenance_prediction:
        sections.append("\nMAINTENANCE RECOMMENDATIONS (AI-GENERATED)\n" + "-"*70)
        
        try:
            # Calculate equipment statistics
            equipment_stats = []
            for equipment in anomalies_df['equipment_id'].unique():
                equip_anomalies = anomalies_df[
                    (anomalies_df['equipment_id'] == equipment) & 
                    (anomalies_df['is_anomaly'])
                ]
                
                stats = {
                    'equipment': equipment,
                    'anomaly_count': len(equip_anomalies),
                }
                
                # Add sensor averages if available
                sensor_cols = ['pressure', 'temperature', 'vibration']
                for col in sensor_cols:
                    if col in equip_anomalies.columns:
                        stats[f'avg_{col}'] = equip_anomalies[col].mean()
                
                equipment_stats.append(stats)
            
            logger.info("Generating maintenance predictions with GenAI...")
            maintenance_pred = genai_service.predict_maintenance_needs(equipment_stats)
            sections.append(maintenance_pred)
            sections.append("")
        except Exception as e:
            logger.error(f"Maintenance prediction failed: {e}")
            sections.append(f"[Maintenance prediction failed: {str(e)}]\n")
    
    return sections


def generate_insight_report_traditional(
    anomalies_df: pd.DataFrame,
    correlated_logs_df: pd.DataFrame,
) -> str:
    """Generate a human-readable insight report using traditional rule-based approach.
    
    This is the original implementation kept for backward compatibility.
    Use generate_insight_report() with config for GenAI support.
    """
    header = (
        "MULTI-MODAL ANOMALY INSIGHT REPORT\n"
        "===================================\n\n"
    )

    summary = _summarise_anomalies(anomalies_df)
    causes = _hypothesise_root_causes(anomalies_df)

    if correlated_logs_df is None or correlated_logs_df.empty:
        corr_section = (
            "No operator logs were strongly correlated with detected anomalies "
            "within the configured time and similarity windows.\n"
        )
    else:
        corr_lines = ["Sample anomaly ↔ log correlations:"]
        # Show top 5 examples
        sample = correlated_logs_df.head(5)
        for _, row in sample.iterrows():
            corr_lines.append(
                f"- {row['anomaly_timestamp']} [{row['anomaly_equipment_id']}] "
                f"<-> log at {row['log_timestamp']}:\n  '{row['log_text']}' "
                f"(similarity={row['similarity']:.2f})"
            )
        corr_section = "\n".join(corr_lines)

    narrative = (
        "\n\nInterpretation (LLM-style narrative, rule-based in this prototype):\n"
        "Based on the detected anomalies and the associated operator logs, the system "
        "highlights equipment with repeated abnormal behaviour and suggests high-level "
        "root cause hypotheses (e.g., pressure spikes, overheating, abnormal vibration). "
        "In a production system, this section would be generated by a large language model "
        "given structured anomaly tables and raw log snippets as context.\n"
    )

    report = "\n\n".join([header, summary, causes, corr_section, narrative])
    return report
