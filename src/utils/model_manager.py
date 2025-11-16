"""
Model versioning and serialization utilities.
"""
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import hashlib
from ..utils.logger import get_logger
from ..utils.exceptions import ModelLoadError, ModelError

logger = get_logger(__name__)


class ModelVersionManager:
    """Manages model versioning, saving, and loading."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelVersionManager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / "model_registry.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load model registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": []}
    
    def _save_metadata(self) -> None:
        """Save model registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Save model with versioning and metadata.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            metadata: Additional metadata to store
            version: Optional version string
            
        Returns:
            Model ID
        """
        try:
            # Generate model ID
            if version:
                model_id = f"{model_name}_{version}"
            else:
                model_id = self._generate_model_id(model_name)
            
            # Save model file
            model_path = self.model_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            
            # Compute model hash
            model_hash = self._compute_model_hash(model_path)
            
            # Prepare metadata
            model_metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "version": version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                "file_path": str(model_path),
                "file_size_bytes": model_path.stat().st_size,
                "model_hash": model_hash,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Update registry
            self.metadata["models"].append(model_metadata)
            self._save_metadata()
            
            logger.info(f"Model saved successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Model save failed: {e}")
    
    def load_model(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ) -> Any:
        """
        Load model by ID, name, or name+version.
        
        Args:
            model_id: Specific model ID to load
            model_name: Model name (loads latest if version not specified)
            version: Version to load for given model_name
            
        Returns:
            Loaded model object
        """
        try:
            # Find model in registry
            model_info = None
            
            if model_id:
                model_info = next(
                    (m for m in self.metadata["models"] if m["model_id"] == model_id),
                    None
                )
            elif model_name:
                candidates = [
                    m for m in self.metadata["models"]
                    if m["model_name"] == model_name
                ]
                
                if version:
                    model_info = next(
                        (m for m in candidates if m["version"] == version),
                        None
                    )
                elif candidates:
                    # Get latest version
                    model_info = max(candidates, key=lambda m: m["created_at"])
            
            if not model_info:
                raise ModelLoadError(
                    f"Model not found: id={model_id}, name={model_name}, version={version}"
                )
            
            # Load model file
            model_path = Path(model_info["file_path"])
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            
            logger.info(f"Model loaded successfully: {model_info['model_id']}")
            return model
            
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model load failed: {e}")
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """
        List all models or filter by name.
        
        Args:
            model_name: Optional filter by model name
            
        Returns:
            List of model metadata dictionaries
        """
        models = self.metadata["models"]
        
        if model_name:
            models = [m for m in models if m["model_name"] == model_name]
        
        return sorted(models, key=lambda m: m["created_at"], reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model by ID.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Find model in registry
            model_info = next(
                (m for m in self.metadata["models"] if m["model_id"] == model_id),
                None
            )
            
            if not model_info:
                logger.warning(f"Model not found in registry: {model_id}")
                return False
            
            # Delete model file
            model_path = Path(model_info["file_path"])
            if model_path.exists():
                model_path.unlink()
            
            # Remove from registry
            self.metadata["models"] = [
                m for m in self.metadata["models"] if m["model_id"] != model_id
            ]
            self._save_metadata()
            
            logger.info(f"Model deleted successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata dictionary or None
        """
        return next(
            (m for m in self.metadata["models"] if m["model_id"] == model_id),
            None
        )
