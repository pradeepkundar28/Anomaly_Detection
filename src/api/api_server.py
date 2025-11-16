"""
FastAPI REST API for anomaly detection.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import run_pipeline
from src.core.detection import detect_anomalies
from src.config import AppConfig, load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.model_manager import ModelVersionManager
from src.utils.validators import validate_sensor_data

# Initialize FastAPI app
app = FastAPI(
    title="Oil Rig Anomaly Detection API",
    description="REST API for multi-modal anomaly detection in oil rig operations",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = setup_logger("api_server", log_dir="logs")

# Initialize model manager
model_manager = ModelVersionManager(model_dir="models")

# Load configuration
try:
    config = load_config("config.yaml")
except:
    config = AppConfig()
    logger.info("Using default configuration")


# Pydantic models
class SensorReading(BaseModel):
    """Single sensor reading."""
    timestamp: str
    equipment_id: str
    sensor_type: str
    value: float


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    sensor_data: List[SensorReading]
    contamination: Optional[float] = Field(default=0.03, ge=0.001, le=0.5)


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    timestamp: str
    equipment_id: str
    anomaly_score: float
    is_anomaly: bool
    sensor_values: Dict[str, float]


class PipelineRequest(BaseModel):
    """Request to run full pipeline."""
    start_date: Optional[str] = "2024-01-01"
    end_date: Optional[str] = "2024-03-31"
    contamination: Optional[float] = 0.03
    output_dir: Optional[str] = "output"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    models_available: int


# API endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Oil Rig Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    models = model_manager.list_models()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models_available=len(models)
    )


@app.post("/api/v1/detect", tags=["Anomaly Detection"])
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest):
    """
    Detect anomalies in provided sensor data.
    
    Args:
        request: Sensor data and detection parameters
        
    Returns:
        List of anomaly detection results
    """
    try:
        logger.info(f"Received anomaly detection request with {len(request.sensor_data)} readings")
        
        # Convert to DataFrame
        sensor_records = [reading.dict() for reading in request.sensor_data]
        sensor_df = pd.DataFrame(sensor_records)
        
        # Validate data
        validate_sensor_data(sensor_df)
        
        # Detect anomalies
        anomalies_df, model = detect_anomalies(
            sensor_df,
            contamination=request.contamination
        )
        
        # Format results
        results = []
        for _, row in anomalies_df[anomalies_df['is_anomaly']].iterrows():
            sensor_values = {
                col: float(row[col])
                for col in anomalies_df.columns
                if col not in ['timestamp', 'equipment_id', 'anomaly_score', 'is_anomaly']
            }
            
            results.append(AnomalyResult(
                timestamp=str(row['timestamp']),
                equipment_id=row['equipment_id'],
                anomaly_score=float(row['anomaly_score']),
                is_anomaly=bool(row['is_anomaly']),
                sensor_values=sensor_values
            ))
        
        logger.info(f"Detected {len(results)} anomalies")
        
        return {
            "status": "success",
            "total_readings": len(sensor_df),
            "anomalies_detected": len(results),
            "anomaly_rate": (len(results) / len(anomalies_df)) * 100,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/pipeline/run", tags=["Pipeline"])
async def run_pipeline_endpoint(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Run the full anomaly detection pipeline.
    
    Args:
        request: Pipeline parameters
        
    Returns:
        Pipeline execution status
    """
    try:
        logger.info("Starting full pipeline execution")
        
        sensor_params = {
            "start": request.start_date,
            "end": request.end_date,
            "freq": "1H"
        }
        
        # Run pipeline in background
        def execute_pipeline():
            try:
                sensor_df, logs_df, anomalies_df, correlated_df, report = run_pipeline(
                    output_dir=request.output_dir,
                    sensor_params=sensor_params,
                    contamination=request.contamination
                )
                logger.info("Pipeline completed successfully")
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        
        background_tasks.add_task(execute_pipeline)
        
        return {
            "status": "started",
            "message": "Pipeline execution started in background",
            "output_dir": request.output_dir
        }
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/upload", tags=["Data"])
async def upload_sensor_data(file: UploadFile = File(...)):
    """
    Upload sensor data CSV file.
    
    Args:
        file: CSV file with sensor data
        
    Returns:
        Upload status and data summary
    """
    try:
        # Read CSV
        contents = await file.read()
        sensor_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate data
        validate_sensor_data(sensor_df)
        
        logger.info(f"Uploaded sensor data: {len(sensor_df)} rows")
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(sensor_df),
            "columns": list(sensor_df.columns),
            "equipment_ids": sensor_df['equipment_id'].unique().tolist(),
            "sensor_types": sensor_df['sensor_type'].unique().tolist(),
            "date_range": {
                "start": str(sensor_df['timestamp'].min()),
                "end": str(sensor_df['timestamp'].max())
            }
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List all available models."""
    try:
        models = model_manager.list_models()
        return {
            "status": "success",
            "count": len(models),
            "models": models
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_id}", tags=["Models"])
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    try:
        model_info = model_manager.get_model_info(model_id)
        
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "status": "success",
            "model": model_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """Delete a model."""
    try:
        success = model_manager.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "status": "success",
            "message": f"Model {model_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/config", tags=["Configuration"])
async def get_configuration():
    """Get current configuration."""
    return {
        "status": "success",
        "config": config.to_dict()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info"
    )
