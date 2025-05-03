import logging
import sys
from pathlib import Path

def setup_logger(name='MyLogger', level=logging.INFO, log_file=None):
    """
    设置并返回一个配置好的 logger 实例。

    Args:
        name (str): logger 的名称。
        level (int): 日志记录级别 (e.g., logging.INFO, logging.DEBUG)。
        log_file (str or Path, optional): 日志输出文件路径。如果提供，日志会同时输出到文件和控制台。

    Returns:
        logging.Logger: 配置好的 logger 实例。
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台输出 Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # 文件输出 Handler (如果指定了 log_file)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False # 防止日志向上传播给 root logger
    return logger

# 可以在项目其他地方这样使用:
# from utils.logger import setup_logger
# logger = setup_logger(log_file=exp_manager.exp_dir / 'logs/experiment.log')
# logger.info("这是一条日志信息")