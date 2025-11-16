"""
Text processing and NLP module.
"""
from .text_processing import (
    build_log_embeddings,
    correlate_logs_with_anomalies
)

__all__ = [
    'build_log_embeddings',
    'correlate_logs_with_anomalies'
]
