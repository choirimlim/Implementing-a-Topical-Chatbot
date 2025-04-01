#!/usr/bin/env python
"""
Script to run the DocuChat API server.
"""

import os
import sys
import logging
import argparse
import yaml
import uvicorn

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DocuChat API server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to config file")
    
    return parser.parse_args()

def main():
    """Main function to run the API server."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    host = args.host or config['api']['host']
    port = args.port or config['api']['port']
    
    # Log startup
    logger.info(f"Starting DocuChat API server on {host}:{port}")
    
    # Run the FastAPI server with Uvicorn
    uvicorn.run(
        "docuchat.api.routes:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
