"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-02-19
@Description：log configuration
==================================================
"""
from loguru import logger
import os
from datetime import datetime

# 1️⃣ Gets the current time (for log directory and file name)
current_time = datetime.now()
log_dir = os.path.join("log", f"{current_time:%Y}", f"{current_time:%m}", f"{current_time:%d}")  # log/年/月/日/
os.makedirs(log_dir, exist_ok=True)  # 自动创建目录

log_file = os.path.join(log_dir, f"{current_time:%H-%M-%S}.log")  # log/YYYY/mm/DD/HH-MM-SS.log

# 2️⃣ Clear the default Loguru configuration (to prevent repeated addition of handlers)
logger.remove()

# 3️⃣ Configure console logs (DEBUG and above, color enabled)
logger.add(
    sink=lambda msg: print(msg, end=""),  # 控制台输出
    level="DEBUG",
    colorize=True
)

# 4️⃣ Configuration file logs (all logs are stored in a unique log file)
logger.add(
    log_file, level="DEBUG", encoding="utf-8"
)

__all__ = ['logger']