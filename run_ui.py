#!/usr/bin/env python
"""
Script to run the DocuChat UI.
"""

import os
import sys
import logging
import argparse
import yaml
import subprocess

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ui_server.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DocuChat UI")
    
    parser.add_argument("--port", type=int, default=8501, help="Port to run the Streamlit UI")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to config file")
    
    return parser.parse_args()

def main():
    """Main function to run the UI."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    port = args.port or config['ui']['port']
    
    # Log startup
    logger.info(f"Starting DocuChat UI on port {port}")
    
    # Run the Streamlit app
    subprocess.run([
        "streamlit", "run", "docuchat/ui/app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()
