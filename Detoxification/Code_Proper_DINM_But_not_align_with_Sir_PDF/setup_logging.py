import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logging():
    """Configure logging with timestamped log files"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'detoxification_{timestamp}.log')
        
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                RotatingFileHandler(
                    log_file,
                    maxBytes=5*1024*1024,  # 5 MB
                    backupCount=3,
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger()
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
        
    except Exception as e:
        print(f"CRITICAL: Failed to initialize logging: {str(e)}")
        # Fallback to basic console logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger()