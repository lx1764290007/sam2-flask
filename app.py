# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import cv2
import torch

from rembg import remove

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

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
cors = CORS(app, supports_credentials=True)

# 加载图像

checkpoint = f"{DEFAULT_CHECK_POINT_PATH}/sam2.1_hiera_base_plus.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
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
    output = remove(np_array)

    image_result = get_mask_to_center_image(None, output)


    # 获取图片尺寸（高度、宽度、通道数）
    height, width, channels = image_result.shape
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

        # 如果你需要将图像保存为 RGBA 格式（带透明度），你可以添加 Alpha 通道
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)  # 创建一个 RGBA 图像
        rgba_image[..., :3] = result_image  # 将 RGB 部分赋值
        rgba_image[..., 3] = (segmentation * 255).astype(np.uint8)  # Alpha 通道根据掩码设置

        # 如果你有处理透明度的函数，可以在这里调用
        rgba_image_center = get_mask_to_center_image(segmentation, rgba_image)

        # 保存图像（假设 save_image 已定义）
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


def generate_mask(np_array, nparr):
    input_point = np.array(np_array)
    new_input_label = np.array([])  # 1 表示前景

    # 使用 OpenCV 解码图像
    image = nparr
    for _ in input_point:
        new_input_label = np.append(new_input_label, 1)  # 添加元素 1

    with torch.inference_mode(), torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, scores = predictor.predict(
            point_coords=input_point,
            point_labels=new_input_label,
            multimask_output=True
        )

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

        # 1. 定义蒙层的颜色和透明度
        overlay_color = (255, 0, 0, 128)  # 红色蒙层，RGBA 格式 (R, G, B, Alpha)
        alpha = overlay_color[3] / 255.0  # 计算透明度比例

        # 2. 创建蒙层
        height, width = image.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[..., 0] = overlay_color[0]  # 红色通道
        overlay[..., 1] = overlay_color[1]  # 绿色通道
        overlay[..., 2] = overlay_color[2]  # 蓝色通道
        overlay[..., 3] = overlay_color[3]  # Alpha 通道

        # 3. 创建与原图相同的 RGBA 图像
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., :3] = image[..., :3]  # 原图 RGB 通道
        rgba_image[..., 3] = (best_mask * 255).astype(np.uint8)  # 使用掩码作为 Alpha 通道
        # 4. 提取掩膜的边缘
        binary_mask = (best_mask * 255).astype(np.uint8)  # 将掩膜转换为二值图像
        edges = cv2.Canny(binary_mask, 100, 200)  # 使用 Canny 边缘检测
        # 5. 创建发光效果
        highlight_color = [173, 216, 230]  # 高亮颜色
        glow_radius = 4  # 高光范围
        # 6. 为掩膜的边缘区域应用发光效果
        for y in range(height):
            for x in range(width):
                if edges[y, x] > 0:  # 如果是边缘
                    rgba_image[y, x, :3] = glow_radius * np.array(highlight_color)  # 将边缘设置为绿色发光
        # 7. 只在掩码部分应用蒙层
        for y in range(height):
            for x in range(width):
                if best_mask[y, x] == 1:  # 如果该像素是掩码区域
                    # 使用 alpha 混合公式将蒙层与原图叠加
                    rgba_image[y, x, :3] = (1 - alpha) * rgba_image[y, x, :3] + alpha * overlay[y, x, :3]
                    rgba_image[y, x, 3] = 255  # 保证蒙层区域完全不透明

        # cv2.waitKey(0)
        # 8. 存储为image并返回文件名
        image_mask = save_image(rgba_image)
        return dict(image_origin=file_name, image_mask=image_mask)


@app.route("/sam2", methods=["POST"])
def sam2() -> tuple[Response, int]:
    torch.cuda.empty_cache()  # 清理未使用的显存
    torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
    # generate_mask(input_point)
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    file_bytes = file.read()
    width = request.form.get("width")  # 图片实际宽度
    height = request.form.get("height")  # 图片实际高度

    np_array = np.frombuffer(file_bytes, np.uint8)
    decoded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    scale_img_result = scale_img(decoded_image, int(width), int(height))  # 图片缩放
    # 将字节流转换为 NumPy 数组

    np_value = request.form.get("np")  # 获取 np 字段
    np_data = json.loads(np_value)
    res = generate_mask(np_data, scale_img_result)
    return jsonify({'data': res}), 200


@app.route('/multiple-sam2', methods=["POST"])
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

    # 转换为 RGB 格式（如果你的模型需要）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    res = create_multiple_masks(image)
    return jsonify({'data': res}), 200

# 自动删除过期图片


@app.route('/rem-bg', methods=["POST"])
def image_remove_bg() -> tuple[Response, int]:
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 解码为 BGR 格式
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # 转换为 RGB 格式（如果你的模型需要）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    res = do_remove_bg(image)
    return jsonify({'data': res}), 200


# 定时删除过期文件
get_scheduler().start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    # get_scheduler().start()
