import os
import time
import logging
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('file_cleaner')

current_dir = Path.cwd()
# 获取当前工作目录

PATH_NAME = 'static'
TARGET_PATH = f"{current_dir}/{PATH_NAME}"

directory = Path(TARGET_PATH)

# 确保目标目录存在
if not directory.exists():
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建目录: {directory}")


# 定义任务：删除超过五分钟的文件
def delete_old_files():
    logger.info("开始清理过期文件...")
    current_time = time.time()  # 当前时间戳
    deleted_count = 0
    
    # 确保目录存在
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return
        
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 文件名格式为 timestamp_随机部分，例如 "1733135032.864875_366"
        if '_' in filename:
            try:
                # 提取文件名中的时间戳部分
                timestamp_str = filename.split('_')[0]
                timestamp = float(timestamp_str)  # 转换为浮动时间戳

                # 计算文件是否超过五分钟
                if current_time - timestamp > 5 * 60:
                    # 超过五分钟，删除文件
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"删除文件: {file_path}")
            except ValueError:
                logger.warning(f"跳过无效文件名: {filename}")
    
    logger.info(f"清理完成，共删除 {deleted_count} 个文件")


def get_scheduler():
    # 创建调度器
    scheduler = BackgroundScheduler()
    
    # 设置每5分钟执行一次清理任务
    scheduler.add_job(delete_old_files, 'interval', minutes=5)
    
    logger.info("文件清理调度器已创建，每5分钟执行一次")
    # 启动调度器
    return scheduler



