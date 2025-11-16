#!/usr/bin/env python3
"""
Convenience script to run the API server from project root.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run
if __name__ == "__main__":
    import uvicorn
    from src.api.api_server import app
    from src.config import load_config
    
    try:
        config = load_config("config.yaml")
    except:
        from src.config import AppConfig
        config = AppConfig()
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info"
    )
