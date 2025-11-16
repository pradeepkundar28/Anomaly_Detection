"""
Configuration management for the anomaly detection system.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import json


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    frequency: str = "1H"
    equipment_ids: List[str] = field(default_factory=lambda: [
        "pump_1", "pump_2", "compressor_1", "compressor_2"
    ])
    sensor_types: List[str] = field(default_factory=lambda: [
        "pressure", "temperature", "vibration"
    ])
    missing_value_rate: float = 0.02
    max_logs_per_day: tuple = (1, 5)


@dataclass
class ModelConfig:
    """Configuration for anomaly detection model."""
    model_type: str = "isolation_forest"
    contamination: float = 0.03
    n_estimators: int = 200
    random_state: int = 42
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1


@dataclass
class TextProcessingConfig:
    """Configuration for text processing and NLP."""
    embedding_type: str = "tfidf"  # 'tfidf' or 'bert'
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    similarity_threshold: float = 0.1
    time_window_hours: int = 2
    top_k_logs: int = 3
    bert_model_name: str = "all-MiniLM-L6-v2"


@dataclass
class OutputConfig:
    """Configuration for output and storage."""
    output_dir: str = "output"
    model_dir: str = "models"
    log_dir: str = "logs"
    save_formats: List[str] = field(default_factory=lambda: ["csv"])
    enable_versioning: bool = True


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    enable_cors: bool = True
    max_content_length: int = 16 * 1024 * 1024  # 16MB


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enable_metrics: bool = True
    metrics_port: int = 8000
    log_predictions: bool = True
    alert_threshold_anomaly_rate: float = 0.1


@dataclass
class GenAIConfig:
    """Configuration for GenAI/LLM features using Ollama."""
    enabled: bool = True  # Enable/disable GenAI features
    provider: str = "ollama"  # Currently only 'ollama' supported
    model: str = "llama3.2:3b"  # Ollama model name
    base_url: str = "http://localhost:11434"  # Ollama API endpoint
    temperature: float = 0.7  # Sampling temperature (0.0-1.0)
    max_tokens: int = 1000  # Maximum tokens to generate
    
    # Feature flags
    enable_anomaly_analysis: bool = True  # Detailed analysis of individual anomalies
    enable_executive_summary: bool = True  # High-level summary generation
    enable_maintenance_prediction: bool = True  # Maintenance recommendations
    
    # Fallback behavior
    fallback_to_traditional: bool = True  # Use rule-based if GenAI fails
    timeout_seconds: int = 120  # Request timeout


@dataclass
class AppConfig:
    """Main application configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    genai: GenAIConfig = field(default_factory=GenAIConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AppConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            AppConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            text_processing=TextProcessingConfig(**config_dict.get('text_processing', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            api=APIConfig(**config_dict.get('api', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'AppConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            AppConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            text_processing=TextProcessingConfig(**config_dict.get('text_processing', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            api=APIConfig(**config_dict.get('api', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'text_processing': asdict(self.text_processing),
            'output': asdict(self.output),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring),
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration
        """
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'text_processing': asdict(self.text_processing),
            'output': asdict(self.output),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring),
        }
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'text_processing': asdict(self.text_processing),
            'output': asdict(self.output),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring),
        }


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        AppConfig instance
    """
    if config_path is None:
        return AppConfig()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        return AppConfig.from_yaml(str(config_path))
    elif config_path.suffix == '.json':
        return AppConfig.from_json(str(config_path))
    else:
        raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
