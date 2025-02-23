import os
import random
from datetime import datetime
from pathlib import Path

from pathlib import Path

from PIL import Image

# 当前目录
current_dir = Path.cwd()

# 获取 app.py 文件的绝对路径

PATH_NAME = 'static'
SAVE_PATH = f"{current_dir}/{PATH_NAME}"

path = Path(SAVE_PATH)


def save_image(np_arr):
    if path.exists():
        image = Image.fromarray(np_arr)
        # 获取当前时间戳
        now = datetime.now()
        timestamp = str(now.timestamp())
        # 生成一个三位数随机数
        random_number = str(random.randrange(100, 1000))
        file_name = f"{SAVE_PATH}/{timestamp}_{random_number}.png"
        try:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            image.save(f"{file_name}", format="PNG")
        except (OSError, IOError) as e:
            # 记录错误信息
            print(f"Error saving image: {e}")
        return f"{timestamp}_{random_number}.png"
    else:
        path.mkdir(parents=True, exist_ok=True)
        save_image(np_arr)
