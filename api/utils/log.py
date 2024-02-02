import os
from datetime import datetime
from loguru import logger
from config import SETTINGS

def init_logger():
    # Create a new log directory with the current server time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(SETTINGS.log_path, current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove any existing handlers, in case this is not the first call
    logger.remove(handler_id=None)
    
    # INFO log file
    logger.add(
        os.path.join(log_dir, 'info.log'), 
        encoding="utf-8", 
        filter=lambda record: record["level"].name == "INFO"
    )

    # DEBUG log file
    logger.add(
        os.path.join(log_dir, 'debug.log'), 
        encoding="utf-8", 
        filter=lambda record: record["level"].name == "DEBUG"
    )

    # ERROR log file
    logger.add(
        os.path.join(log_dir, 'error.log'), 
        encoding="utf-8", 
        filter=lambda record: record["level"].name == "ERROR"
    )

    # CRITICAL log file
    logger.add(
        os.path.join(log_dir, 'critical.log'), 
        encoding="utf-8", 
        filter=lambda record: record["level"].name == "CRITICAL"
    )
    
    logger.info("Logger initialized in new directory: {}", log_dir)
    return logger
