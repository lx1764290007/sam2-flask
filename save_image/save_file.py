import base64
import io
from PIL import Image
import numpy as np


def image_to_base64(np_arr: np.ndarray) -> str:
    """
    将numpy数组格式的图像转换为base64编码字符串
    
    Args:
        np_arr (np.ndarray): 输入的图像数组，支持RGB或RGBA格式
        
    Returns:
        str: base64编码的图像字符串，格式为 "data:image/png;base64,{encoded_data}"
        
    Raises:
        ValueError: 当输入参数无效时
        TypeError: 当输入类型不正确时
    """
    # 输入参数验证
    if not isinstance(np_arr, np.ndarray):
        raise TypeError("输入参数必须是numpy数组")
    
    if np_arr.size == 0:
        raise ValueError("输入的图像数组不能为空")
    
    if len(np_arr.shape) not in [2, 3]:
        raise ValueError("图像数组必须是2D或3D格式")
    
    try:
        # 将numpy数组转换为PIL图像
        if len(np_arr.shape) == 2:
            # 灰度图像
            image = Image.fromarray(np_arr, mode='L')
        elif np_arr.shape[2] == 3:
            # RGB图像
            image = Image.fromarray(np_arr, mode='RGB')
        elif np_arr.shape[2] == 4:
            # RGBA图像
            image = Image.fromarray(np_arr, mode='RGBA')
        else:
            raise ValueError(f"不支持的通道数: {np_arr.shape[2]}")
        
        # 使用BytesIO保存图像到内存
        buffer = io.BytesIO()
        
        # 根据图像模式选择合适的格式
        if image.mode in ['RGBA', 'LA']:
            image.save(buffer, format='PNG')
            mime_type = 'image/png'
        else:
            image.save(buffer, format='JPEG', quality=95)
            mime_type = 'image/jpeg'
        
        # 获取图像字节数据
        image_bytes = buffer.getvalue()
        
        # 转换为base64编码
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        
        # 返回完整的data URL格式
        return f"data:{mime_type};base64,{base64_encoded}"
        
    except Exception as e:
        raise RuntimeError(f"图像处理过程中发生错误: {str(e)}")


def save_image(np_arr: np.ndarray) -> str:
    """
    保持向后兼容性的包装函数
    
    Args:
        np_arr (np.ndarray): 输入的图像数组
        
    Returns:
        str: base64编码的图像字符串
    """
    return image_to_base64(np_arr)
