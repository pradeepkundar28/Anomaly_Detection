"""
Data validation utilities.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from .exceptions import DataValidationError
from .logger import get_logger

logger = get_logger(__name__)


def validate_sensor_data(df: pd.DataFrame) -> None:
    """
    Validate sensor data DataFrame structure and content.
    
    Args:
        df: Sensor data DataFrame
        
    Raises:
        DataValidationError: If validation fails
    """
    required_columns = ['timestamp', 'equipment_id', 'sensor_type', 'value']
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )
    
    # Check for empty DataFrame
    if df.empty:
        raise DataValidationError("Sensor data DataFrame is empty")
    
    # Check timestamp format
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise DataValidationError(f"Invalid timestamp format: {e}")
    
    # Check for null values in critical columns
    null_counts = df[['timestamp', 'equipment_id', 'sensor_type']].isnull().sum()
    if null_counts.any():
        raise DataValidationError(
            f"Null values found in critical columns: {null_counts[null_counts > 0].to_dict()}"
        )
    
    # Check value column data type
    if not pd.api.types.is_numeric_dtype(df['value']):
        raise DataValidationError("'value' column must be numeric")
    
    # Check for duplicate timestamps per equipment-sensor
    duplicates = df.groupby(['timestamp', 'equipment_id', 'sensor_type']).size()
    if (duplicates > 1).any():
        n_duplicates = (duplicates > 1).sum()
        logger.warning(f"Found {n_duplicates} duplicate timestamp-equipment-sensor combinations")
    
    logger.info(f"Sensor data validation passed: {len(df)} rows, "
                f"{df['equipment_id'].nunique()} equipment, "
                f"{df['sensor_type'].nunique()} sensor types")


def validate_operator_logs(df: pd.DataFrame) -> None:
    """
    Validate operator logs DataFrame structure and content.
    
    Args:
        df: Operator logs DataFrame
        
    Raises:
        DataValidationError: If validation fails
    """
    required_columns = ['timestamp', 'equipment_id', 'log_text']
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )
    
    # Check for empty DataFrame
    if df.empty:
        logger.warning("Operator logs DataFrame is empty")
        return
    
    # Check timestamp format
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise DataValidationError(f"Invalid timestamp format: {e}")
    
    # Check for empty log texts
    empty_logs = df['log_text'].isna() | (df['log_text'] == '')
    if empty_logs.any():
        n_empty = empty_logs.sum()
        logger.warning(f"Found {n_empty} empty log texts")
    
    logger.info(f"Operator logs validation passed: {len(df)} logs")


def validate_anomaly_results(df: pd.DataFrame) -> None:
    """
    Validate anomaly detection results DataFrame.
    
    Args:
        df: Anomaly results DataFrame
        
    Raises:
        DataValidationError: If validation fails
    """
    required_columns = ['timestamp', 'equipment_id', 'anomaly_score', 'is_anomaly']
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )
    
    # Check for empty DataFrame
    if df.empty:
        raise DataValidationError("Anomaly results DataFrame is empty")
    
    # Check anomaly_score is numeric
    if not pd.api.types.is_numeric_dtype(df['anomaly_score']):
        raise DataValidationError("'anomaly_score' must be numeric")
    
    # Check is_anomaly is boolean
    if not pd.api.types.is_bool_dtype(df['is_anomaly']):
        raise DataValidationError("'is_anomaly' must be boolean")
    
    # Check for anomalies detected
    n_anomalies = df['is_anomaly'].sum()
    anomaly_rate = (n_anomalies / len(df)) * 100
    
    if n_anomalies == 0:
        logger.warning("No anomalies detected in the dataset")
    
    logger.info(f"Anomaly results validation passed: {n_anomalies} anomalies "
                f"({anomaly_rate:.2f}%) out of {len(df)} data points")


def check_data_quality(df: pd.DataFrame, column: str = 'value') -> dict:
    """
    Check data quality metrics for a numerical column.
    
    Args:
        df: DataFrame to check
        column: Column name to analyze
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'missing_count': df[column].isna().sum(),
        'missing_percentage': (df[column].isna().sum() / len(df)) * 100,
        'zero_count': (df[column] == 0).sum(),
        'negative_count': (df[column] < 0).sum(),
        'infinite_count': np.isinf(df[column]).sum() if pd.api.types.is_numeric_dtype(df[column]) else 0,
        'mean': df[column].mean() if pd.api.types.is_numeric_dtype(df[column]) else None,
        'std': df[column].std() if pd.api.types.is_numeric_dtype(df[column]) else None,
        'min': df[column].min() if pd.api.types.is_numeric_dtype(df[column]) else None,
        'max': df[column].max() if pd.api.types.is_numeric_dtype(df[column]) else None,
    }
    
    return metrics
