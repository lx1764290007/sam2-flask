import modal
from app import app as flask_app

image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "libgl1",  # OpenCV 图形相关
        "libglib2.0-0",  # OpenCV 多线程相关（libgthread 所属包）
    )
    .run_commands(
        "git clone https://github.com/lx1764290007/sam2-flask.git /root/sam"
    )
    .pip_install("gunicorn")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("checkpoints/sam2.1_hiera_base_plus.pt", "/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                    copy=True)  # 模型
)
app = modal.App(name="sam-app-3", image=image)


@app.function(
    gpu="T4",  # 可选: "A100", "T4", "A10G"
    image=image,
    cpu=4,
    memory=1024 * 16,
    timeout=60*10
)
@modal.web_server(port=10088)  # 开启 Web 服务（Flask/FastAPI）
def web():
    return flask_app  # 直接返回 Flask 实例


