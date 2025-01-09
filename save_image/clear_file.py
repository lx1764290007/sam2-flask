import os
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

current_dir = Path.cwd()
# 获取 app.py 文件的绝对路径

PATH_NAME = 'static'
TARGET_PATH = f"{current_dir}/{PATH_NAME}"

directory = Path(TARGET_PATH)


# 定义任务：删除超过五分钟的文件
def delete_old_files():
    current_time = time.time()  # 当前时间戳
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
                    print(f"删除文件: {file_path}")
            except ValueError:
                print(f"跳过无效文件名: {filename}")


def get_scheduler():
    # 创建调度器
    scheduler = BackgroundScheduler()
    scheduler.add_job(delete_old_files, 'cron', hour=0, minute=0)

    # 启动调度器
    return scheduler



