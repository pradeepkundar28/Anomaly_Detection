"""
Custom exceptions for the anomaly detection system.
"""


class AnomalyDetectionError(Exception):
    """Base exception for all anomaly detection errors."""
    pass


class DataValidationError(AnomalyDetectionError):
    """Raised when data validation fails."""
    pass


class ModelError(AnomalyDetectionError):
    """Raised when model operations fail."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when attempting to use an untrained model."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ConfigurationError(AnomalyDetectionError):
    """Raised when configuration is invalid."""
    pass


class DataProcessingError(AnomalyDetectionError):
    """Raised when data processing fails."""
    pass


class TextProcessingError(AnomalyDetectionError):
    """Raised when text processing fails."""
    pass


class PipelineError(AnomalyDetectionError):
    """Raised when pipeline execution fails."""
    pass
