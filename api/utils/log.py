import os
import sys
import logging
import inspect
from datetime import datetime
from loguru import logger
from config import SETTINGS

LOG_PATH = SETTINGS.log_path
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
CONSOLE_FORMAT = '<level>{level: <8}</level>  <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'
FILE_FORMAT = '{level: <8}  {time:YYYY-MM-DD HH:mm:ss.SSS} - {name}:{function} - {message}'

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe().f_back
        depth = 0
        while frame and logging.__file__ in frame.f_code.co_filename:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def init_logger(level: str = "INFO", log_path: str = LOG_PATH, show_console: bool = True):

    logger.remove()  # 移除Loguru的默认处理器
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_path, current_time)

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create log directory: {e}")
        raise

    # 拦截所有标准日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("uvicorn.access").propagate = False

    # 控制台日志
    if show_console:
        logger.add(sys.stdout, level=level, format=CONSOLE_FORMAT)

    # 文件日志
    for log_level in LOG_LEVELS:
        logger.add(
            os.path.join(log_dir, f'{log_level.lower()}.log'),
            encoding="utf-8",
            level=log_level,
            filter=lambda record, lvl=log_level: record["level"].name == lvl,
            format=FILE_FORMAT,
            rotation="10 MB",
            retention="7 days"
        )

    logger.info("Logger initialized in directory: {}", log_dir)
    return logger