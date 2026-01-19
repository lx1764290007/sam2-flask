# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import time
# 为 SAM 2 设置依赖项
from pathlib import Path
from werkzeug.utils import secure_filename
import modal
import sys

import json
import cv2
import torch
import io
import base64

from rembg import remove, new_session
from PIL import Image

import os
import random
from datetime import datetime

from sam2.build_sam import build_sam2
import numpy as np

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2 import automatic_mask_generator

from app_conf import (
    DEVICE_TYPE,
    DEFAULT_CHECK_POINT_PATH,
)

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from save_image.clear_file import get_scheduler
from save_image.save_file import save_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append("/root/flask-server2")  # 👈 把 server 目录加入 Python 路径
sys.path.append("/root/sam2_configs")
sys.path.append("/root/flask-server2/sam2")
sys.path.append("/root/flask-server2/checkpoints")
sys.path.insert(0, "/root/flask-server2/sam2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "CHECK_POINT_PATH": "/root/flask-server2/checkpoints",
        "CHECK_POINT_CONFIG_PATH": "/root/flask-server2/configs",
        "DEFAULT_CHECK_POINT_PATH": "/root/flask-server2/checkpoints",
        "DATA_PATH": "/root/flask-server2",
        "APP_ROOT": "/root/flask-server2",
        "DEVICE_TYPE": "cuda",
        "U2NET_HOME": "/root/.u2net",
    })
    .apt_install("git", "wget", "python3-opencv", "ffmpeg",
                 "libgl1",  # OpenCV 图形相关
                 "libglib2.0-0",  # OpenCV 多线程相关（libgthread 所属包）
                 )
    .pip_install(
        "torch~=2.4.1",
        "torchvision==0.19.1 ",
        "opencv-python==4.10.0.84",
        "onnxruntime-gpu",
        "onnx==1.17.0",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("app_conf", "sam2", "save_image", copy=True)
    .add_local_file("sam2/sam2.1_hiera_b+.yaml", "/root/sam2/sam2.1_hiera_b+.yaml", copy=True)
    .add_local_file("checkpoints/sam2.1_hiera_base_plus.pt",
                    "/root/flask-server2/checkpoints/sam2.1_hiera_base_plus.pt", copy=True)
    .add_local_file("app_conf.py", "/root/flask-server2/app_conf.py", copy=True)
    .add_local_file("app.py", "/root/flask-server2/app.py", copy=True)
    .add_local_file("sam2/build_sam.py", "/root/flask-server2/sam2/build_sam.py", copy=True)
    .add_local_dir("sam2/configs", "/root/flask-server2/configs", copy=True)
    .add_local_dir("sam2/configs", "/root/flask-server2/checkpoints/configs", copy=True)
    .add_local_dir("sam2_configs", "/root/sam2_configs", copy=True)
    .add_local_dir("sam2_configs", "/root/flask-server2/sam2_configs", copy=True)
)
app_name = "sam2-app"
app = modal.App(app_name, image=image)

# 将使用 modal Volume 缓存模型权重，这样在启动新容器时就不需要重复下载它们。
video_vol = modal.Volume.from_name("sam2-inputs", create_if_missing=True)
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
rembg_cache_vol = modal.Volume.from_name("rembg-cache", create_if_missing=True)  # 添加 rembg 模型缓存卷
cache_dir = "/cache"

current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_NAME = 'static'
SAVE_PATH = f"{current_dir}/{PATH_NAME}"
path = Path(SAVE_PATH)

# 初始化支持 GPU 的 session，使用环境变量配置的缓存目录
session = new_session("u2net", use_cuda=False)  # 使用 u2net 模型并启用 CUDA

logger = logging.getLogger(__name__)

app_flask = Flask(__name__)
app_flask.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
cors = CORS(app_flask, supports_credentials=True)

# 加载图像

checkpoint = f"{DEFAULT_CHECK_POINT_PATH}/sam2.1_hiera_base_plus.pt"
model_cfg = "sam2.1_hiera_b+.yaml"
build_sam2_model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(build_sam2_model)

mask_generator = automatic_mask_generator.SAM2AutomaticMaskGenerator(model=build_sam2_model)


# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式
# 提供一个前景点，格式为 [x, y]（像素坐标）
# input_point = np.array([[200, 100]])


def create_multiple_masks(nd):
    masks = mask_generator.generate(image=nd)
    return multiple_masks_to_image(masks, nd)


def do_remove_bg(np_array):
    output = remove(np_array, session=session)
    image_result = get_mask_to_center_image(None, output)
    # 获取图片尺寸（高度、宽度、通道数）
    # height, width, channels = image_result.shape
    result = save_image(image_result)
    return result


def multiple_masks_to_image(masks, np_arr):
    arr = []
    for i, mask in enumerate(masks):
        # 从字典中提取实际的掩码数据
        segmentation = mask.get('segmentation')
        if segmentation is None:
            raise ValueError(f"Mask {i} does not contain 'segmentation' key")

        # 获取图像的高度和宽度
        height, width = np_arr.shape[:2]

        # 将掩码扩展为 3 通道的掩码
        mask_3d = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)  # 转换为 3 通道（RGB）

        # 将掩码应用到原图上（掩码为 1 的部分保留，掩码为 0 的部分置为透明或黑色）
        result_image = np_arr * mask_3d  # 根据掩码提取原图区域

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)  # 创建一个 RGBA 图像
        rgba_image[..., :3] = result_image  # 将 RGB 部分赋值
        rgba_image[..., 3] = (segmentation * 255).astype(np.uint8)  # Alpha 通道根据掩码设置

        rgba_image_center = get_mask_to_center_image(segmentation, rgba_image)

        # 保存图像
        file_name = save_image(rgba_image_center)

        # 将文件名添加到数组
        arr.append(file_name)
    return arr


# 图片缩放
def scale_img(img, width, height):
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)


def resize_image_aspect_ratio(image, target_width=None, target_height=None):
    """
    等比缩放图片到指定宽度或高度。

    :param image: 输入的图片 (numpy array)
    :param target_width: 目标宽度（可选）
    :param target_height: 目标高度（可选）
    :return: 等比缩放后的图片
    """
    height, width = image.shape[:2]

    if target_width is None and target_height is None:
        raise ValueError("必须提供目标宽度 (target_width) 或目标高度 (target_height)")

    # 计算缩放比例
    if target_width is not None:
        scale = target_width / width
    else:
        scale = target_height / height

    # 计算新的宽高
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 使用 OpenCV 进行缩放
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


# 把抠图绘制到透明画布中间
def get_mask_to_center_image(best_mask, rgba_image):
    # 图片尺寸
    height, width, _ = rgba_image.shape
    # 1. 提取非透明部分的边界
    alpha_channel = rgba_image[..., 3]
    non_zero_coords = np.where(alpha_channel > 0)
    if non_zero_coords[0].size == 0 or non_zero_coords[1].size == 0:
        raise ValueError("掩膜区域为空，无法裁剪")

    y_min, y_max = non_zero_coords[0].min(), non_zero_coords[0].max()
    x_min, x_max = non_zero_coords[1].min(), non_zero_coords[1].max()

    # 裁剪非透明部分
    cropped_image = rgba_image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_height, cropped_width, _ = cropped_image.shape
    # 2. 创建目标透明图片
    target_height, target_width = max(cropped_height, cropped_width), max(cropped_height, cropped_width)  # 自定义目标图片大小
    transparent_image = np.zeros((target_height, target_width, 4), dtype=np.uint8)

    # 3. 计算非透明部分在目标图片中的中心位置
    # cropped_height, cropped_width, _ = cropped_image.shape
    # if cropped_height < cropped_width < rgba_image_size:
    #     cropped_image = resize_image_aspect_ratio(cropped_image, target_width=mask_size_width)
    #     cropped_height, cropped_width, _ = cropped_image.shape
    # elif cropped_width < cropped_height < rgba_image_size:
    #     cropped_image = resize_image_aspect_ratio(cropped_image, target_height=mask_size_height)
    #     cropped_height, cropped_width, _ = cropped_image.shape
    # cropped_image = resize_image_aspect_ratio(cropped_image, target_height=max(mask_size_width, mask_size_height))
    # cropped_height, cropped_width, _ = cropped_image.shape
    start_y = (target_height - cropped_height) // 2
    start_x = (target_width - cropped_width) // 2

    # 4. 将裁剪后的图像放置到目标透明图片的中心
    transparent_image[start_y:start_y + cropped_height, start_x:start_x + cropped_width] = cropped_image
    return transparent_image


# 封裝遠程方法generate_mask
@app.function(
    volumes={"/root/.u2net": rembg_cache_vol},
    gpu="T4",
    max_containers=2,
    # scaledown_window=60 * 60,
    min_containers=1,
    scaledown_window=3600,  # 最大值，作为额外保障
    enable_memory_snapshot=True,  # 快速恢复状态
    cpu=2.0
)
def torchFunc(img=None, input_point=None, new_input_label=None):
    if img is None:
        return f"读取文件时发生错误"
    with torch.inference_mode(), torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
        predictor.set_image(img)
        return predictor.predict(
            point_coords=input_point,
            point_labels=new_input_label,
            multimask_output=True
        )


def generate_mask(np_array, nparr):
    torchFunc = modal.Function.from_name(app_name, "torchFunc")
    try:
        input_point = np.array(np_array)
        new_input_label = np.array([])  # 1 表示前景

        # 使用 OpenCV 解码图像
        image = nparr
        for _ in input_point:
            new_input_label = np.append(new_input_label, 1)  # 添加元素 1

        masks, _, scores = torchFunc.remote(image, input_point, new_input_label)

        # 扁平化的线性索引
        flat_best_mask_index = np.argmax(scores)  # 返回的是一个扁平化的线性索引

        # 将其转换为 (掩码索引, 高度, 宽度) 的索引
        best_mask_index = flat_best_mask_index // (scores.shape[1] * scores.shape[2])  # 获取掩码的索引
        # y = (flat_best_mask_index % (scores.shape[1] * scores.shape[2])) // scores.shape[2]  # 获取高度索引
        # x = flat_best_mask_index % scores.shape[2]  # 获取宽度索引

        # 选择得分最高的掩码
        # 获取对应的掩码
        best_mask = masks[best_mask_index]
        height, width = best_mask.shape
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        # 确保 RGB 通道顺序正确
        rgba_image[..., 0:3] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用掩膜作为 Alpha 通道
        rgba_image[..., 3] = (best_mask * 255).astype(np.uint8)

        rgba_image_center = get_mask_to_center_image(best_mask, rgba_image)

        # 缩放到 300 * 300
        # image_center_scale2_300_300 = scale_img(rgba_image_center, 500, 500)
        # 将边缘设置为高亮
        file_name = save_image(rgba_image_center)
        # 保存为 PNG 并转 Base64
        # buffer = BytesIO()
        # Image.fromarray(image_center_scale2_300_300).save(buffer, format="PNG")
        # image_base64_result = base64.b64encode(buffer.getvalue()).decode('utf-8')
        #
        # # 使用 base64 编码字节流
        # base64_image_origin = f"data:image/png;base64,{image_base64_result}"

        # 定义蒙层的颜色和透明度
        overlay_color = (255, 0, 0, 128)  # 红色蒙层，RGBA 格式 (R, G, B, Alpha)
        alpha = overlay_color[3] / 255.0  # 计算透明度比例

        # 创建蒙层
        height, width = image.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[..., 0] = overlay_color[0]  # 红色通道
        overlay[..., 1] = overlay_color[1]  # 绿色通道
        overlay[..., 2] = overlay_color[2]  # 蓝色通道
        overlay[..., 3] = overlay_color[3]  # Alpha 通道

        # 创建与原图相同的 RGBA 图像
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., :3] = image[..., :3]  # 原图 RGB 通道
        rgba_image[..., 3] = (best_mask * 255).astype(np.uint8)  # 使用掩码作为 Alpha 通道
        # 提取掩膜的边缘
        binary_mask = (best_mask * 255).astype(np.uint8)  # 将掩膜转换为二值图像
        edges = cv2.Canny(binary_mask, 100, 200)  # 使用 Canny 边缘检测
        highlight_color = [173, 216, 230]  # 高亮颜色
        glow_radius = 4  # 高光范围
        for y in range(height):
            for x in range(width):
                if edges[y, x] > 0:  # 如果是边缘
                    rgba_image[y, x, :3] = glow_radius * np.array(highlight_color)
        # 只在掩码部分应用蒙层
        for y in range(height):
            for x in range(width):
                if best_mask[y, x] == 1:  # 如果该像素是掩码区域
                    # 使用 alpha 混合公式将蒙层与原图叠加
                    rgba_image[y, x, :3] = (1 - alpha) * rgba_image[y, x, :3] + alpha * overlay[y, x, :3]
                    rgba_image[y, x, 3] = 255  # 保证蒙层区域完全不透明

        # cv2.waitKey(0)
        image_mask = save_image(rgba_image)
        return {"data": {"image_origin": file_name, "image_mask": image_mask}, "status": "success", "code": 200}
    except Exception as e:
        logger.error(f"生成掩膜時出錯: {str(e)}", exc_info=True)
        return {"message": f"Processing failed: {str(e)}", "status": "error"}


current_timestamp = time.time()
timestamp_record_file_path = "record.txt"
try:
    with open(timestamp_record_file_path, 'w', encoding='utf-8') as file:
        file.write(str(current_timestamp))
except EOFError as e:
    logger.error(f"打开文件: {timestamp_record_file_path}出错", exc_info=True)


@app_flask.route("/pull-up", methods=["POST", "GET"])
def pull_up_app() -> tuple[Response, int]:

    rem_bg_remote = modal.Function.from_name(app_name, "rem_remote")
    torch_func_remote = modal.Function.from_name(app_name, "torchFunc")
    global current_timestamp
    try:

        with open(timestamp_record_file_path, 'r', encoding='utf-8') as current_file:

            first_line = current_file.readline()
            print(str(first_line))
            current_time = first_line or str(time.time())
            if time.time() - float(current_time) > 10:
                rem_bg_remote.remote(None)
                torch_func_remote.remote(None)
                try:
                    with open(timestamp_record_file_path, 'w', encoding='utf-8') as origin_file:
                        origin_file.write(str(time.time()))
                except EOFError as e:
                    logger.error(f"打开文件: {timestamp_record_file_path},{str(e)}出错", exc_info=True)
    except Exception as e:
        return jsonify({
            "data": e,
            "code": 200
        }), 200
    return jsonify({
        "data": "success",
        "code": 200
    }), 200


# 单图定點抠图
@app_flask.route("/sam2", methods=["POST"])
def sam2() -> tuple[Response, int]:
    # torch.cuda.empty_cache()  # 清理未使用的显存
    # torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
    # generate_mask(input_point)
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    width = request.form.get("width")  # 图片实际宽度
    height = request.form.get("height")  # 图片实际高度
    np_value = request.form.get("np")  # 获取 np 字段

    try:
        file_bytes = file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        decoded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        scale_img_result = scale_img(decoded_image, int(width), int(height))  # 图片缩放
        # 将字节流转换为 NumPy 数组
        np_data = json.loads(np_value)
        # result = sam2_remote.remote(np_data,scale_img_result)

        result = generate_mask(np_data, scale_img_result)

        if result.get("status") == "error":
            return jsonify({"error": result.get("message", "Unknown error")}), 400
        else:
            # 获取返回的数据（包含base64编码图像数据的字典）
            data = result.get("data")

            # 处理字典中的每个base64图像数据
            processed_data = {}
            for key, value in data.items():
                try:
                    filename, error = process_base64_image(value)
                    if error:
                        return jsonify({"error": f"处理图像 {key} 失败: {error}"}), 500
                    processed_data[key] = filename
                except Exception as e:
                    error_msg = f"处理图像 {key} 失败: {str(e)}"
                    logging.error(error_msg)
                    return jsonify({"error": error_msg}), 500

            # 返回处理后的数据
            return jsonify({
                "data": processed_data
            }), 200
    except Exception as e:
        error_msg = f"远程处理失败: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500


# 多圖抠圖
@app_flask.route('/multiple-sam2', methods=["POST"])
def create_multiple_images() -> tuple[Response, int]:
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    file_bytes = file.read()
    width = request.form.get("width")  # 图片实际宽度
    height = request.form.get("height")  # 图片实际高度
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # 解码为 BGR 格式
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    res = create_multiple_masks(image)
    return jsonify({'data': res}), 200


# 确保static目录存在
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    logger.info(f"创建目录: {SAVE_PATH}")
else:
    logger.info(f"使用现有目录: {SAVE_PATH}")


@app_flask.route('/rem-bg', methods=["POST"])
def image_remove_bg() -> tuple[Response, int]:
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file_bytes = file.read()
        rem_bg_remote = modal.Function.from_name(app_name, "rem_remote")
        result = rem_bg_remote.remote(file_bytes)

        if result.get("status") == "error":
            image = Image.open(io.BytesIO(file_bytes))
            # 获取当前时间戳
            now = datetime.now()
            timestamp = str(now.timestamp())
            # 生成一个三位数随机数
            random_number = str(random.randrange(100, 1000))
            filename = f"{timestamp}_{random_number}.png"
            file_path = f"{SAVE_PATH}/{filename}"

            # 确保图像是RGBA模式
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # 保存图像
            image.save(file_path, format="PNG")

            return jsonify({
                "data": filename,
                "message": str(result.get("status"))
            }), 200
        else:
            data = result.get("data")
            # 将base64数据转换为图片
            try:
                filename, error = process_base64_image(data)
                if error:
                    return jsonify({"error": error}), 500

                return jsonify({
                    "data": filename
                }), 200
            except Exception as e:
                # 记录详细错误信息
                error_msg = f"远程处理失败: {str(e)}"
                logging.error(error_msg)
                return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"远程处理失败: {str(e)}"
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500


# 封裝遠程方法
@app.function(
    max_containers=2,
    # scaledown_window=60 * 60,
    min_containers=1,
    scaledown_window=3600,  # 最大值，作为额外保障
    enable_memory_snapshot=True,  # 快速恢复状态
    nonpreemptible=True,  # 使用非抢占式实例，避免被回收
    cpu=4.0,
    volumes={"/root/.u2net": rembg_cache_vol},  # 将 rembg 缓存卷映射到模型目录
    # gpu="T4",  # 为背景移除任务分配GPU
)
def rem_remote(file_bytes=None) -> dict:
    if not file_bytes:


        return {"message": "No file data provided", "status": "error"}
    try:
        # 验证 rembg 模型文件是否存在
        import os
        u2net_model_path = "/root/.u2net/u2net.onnx"
        if not os.path.exists(u2net_model_path):
            # 如果模型不存在，尝试重新初始化
            try:
                from rembg import new_session
                new_session("u2net", use_cuda=False)
            except Exception as model_error:
                return {"message": f"Failed to initialize rembg model: {str(model_error)}", "status": "error"}

        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 解码为 BGR 格式
        if image is None:
            return {"message": "Failed to decode image", "status": "error"}

        # 转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = do_remove_bg(image)
        return {"data": res, "status": "success"}
    except Exception as e:

        return {"data": process_base64_image(file_bytes), "status": "success", "msg": str(e)}
        # return {"message": f"Processing failed: {str(e)}", "status": "error"}


# 定时删除过期文件
get_scheduler().start()


@app.local_entrypoint()
def main():
    logger.info("启动 SAM2 Flask 服务器...")
    logger.info(f"图片将保存在: {SAVE_PATH}")
    logger.info("文件清理任务已启动，每5分钟执行一次")
    app_flask.run(host="0.0.0.0", port=5000)


def run_flask():
    """Run the Flask development server"""
    logger.info("启动 SAM2 Flask 服务器...")
    logger.info(f"图片将保存在: {SAVE_PATH}")
    logger.info("文件清理任务已启动，每5分钟执行一次")
    app_flask.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run_flask()


# 在文件中添加新的函数，放在适当的位置（在其他辅助函数附近）
def process_base64_image(data):
    """
    处理base64编码的图像数据，将其转换为图像文件并保存

    Args:
        data: base64编码的图像数据

    Returns:
        tuple: (filename, None) 成功时返回文件名，或 (None, error_msg) 失败时返回错误信息
    """
    try:
        # 检查是否包含data:image格式的前缀，如果有则去除
        if data.startswith('data:image'):
            # 找到base64数据的实际开始位置
            base64_start = data.find('base64,')
            if base64_start != -1:
                # 跳过'base64,'这7个字符
                data = data[base64_start + 7:]

        image = base64.b64decode(data)
        image = Image.open(io.BytesIO(image))

        # 获取当前时间戳
        now = datetime.now()
        timestamp = str(now.timestamp())
        # 生成一个三位数随机数
        random_number = str(random.randrange(100, 1000))
        filename = f"{timestamp}_{random_number}.png"
        file_path = f"{SAVE_PATH}/{filename}"

        # 确保图像是RGBA模式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # 保存图像
        image.save(file_path, format="PNG")

        return filename, None
    except Exception as e:
        # 记录详细错误信息
        error_msg = f"图像处理错误: {str(e)}"
        logging.error(error_msg)
        return None, error_msg
