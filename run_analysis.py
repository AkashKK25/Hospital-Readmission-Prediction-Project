#!/usr/bin/env python
"""
Hospital Readmission Prediction Project
Main script to run the full analysis pipeline
"""

import os
import logging
import argparse
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(args):
    """Main function to run the full analysis pipeline"""
    start_time = time.time()
    
    try:
        # Step 1: Data Acquisition
        logger.info("Step 1: Data Acquisition")
        from src.data.acquisition import main as acquisition_main
        acquisition_main()
        
        # Step 2: Data Preprocessing
        if not args.skip_preprocessing:
            logger.info("Step 2: Data Preprocessing")
            from src.data.preprocessing import main as preprocessing_main
            preprocessing_main()
        
        # Step 3: Model Training
        if not args.skip_training:
            logger.info("Step 3: Model Training")
            from src.models.train_model import main as train_main
            train_main()
        
        # Step 4: Business Impact Analysis
        if not args.skip_business_analysis:
            logger.info("Step 4: Business Impact Analysis")
            from src.analysis.business_analysis import main as business_main
            business_main()
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Pipeline completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Launch dashboard if requested
        if args.launch_dashboard:
            logger.info("Launching interactive dashboard...")
            try:
                import subprocess
                subprocess.Popen(["python", "dashboard/app.py"])
                logger.info("Dashboard launched! Access it at http://localhost:8050")
            except Exception as e:
                logger.error(f"Failed to launch dashboard: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Hospital Readmission Analysis Pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-business-analysis", action="store_true", help="Skip business impact analysis step")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch the interactive dashboard after completion")
    
    args = parser.parse_args()
    
    exit_code = main(args)
    exit(exit_code)
