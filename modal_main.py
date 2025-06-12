import os

import modal
import sys

sys.path.append("/root/flask-server2")  # 👈 把 server 目录加入 Python 路径
sys.path.append("/root/sam2_configs")
sys.path.append("/root/flask-server2/sam2")
sys.path.append("/root/flask-server2/checkpoints")
sys.path.insert(0, "/root/flask-server2/sam2")


image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "libgl1",  # OpenCV 图形相关
        "libglib2.0-0",  # OpenCV 多线程相关（libgthread 所属包）
    )
    .run_commands(
        "git clone https://github.com/lx1764290007/sam2-flask.git /root/flask-server2"
    )
    .pip_install("gunicorn")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("sam2/sam2.1_hiera_b+.yaml", "/root/flask-server2/sam2/sam2.1_hiera_b+.yaml", copy=True)
    .add_local_file("checkpoints/sam2.1_hiera_base_plus.pt", "/root/flask-server2/checkpoints/sam2.1_hiera_base_plus.pt", copy=True)  # 模型
    .add_local_file("app_conf.py", "/root/flask-server2/app_conf.py", copy=True)
    .add_local_file("app.py", "/root/flask-server2/app.py", copy=True)
    .add_local_file("sam2/build_sam.py", "/root/flask-server2/sam2/build_sam.py", copy=True)
    .add_local_dir("sam2/configs", "/root/flask-server2/configs")
    .add_local_dir("sam2/configs", "/root/flask-server2/checkpoints/configs")
    .add_local_dir("sam2_configs", "/root/sam2_configs")
    .add_local_dir("sam2_configs", "/root/flask-server2/sam2_configs")
)
app = modal.App(name="sam-web-app3", image=image)


@app.function(
    gpu="A100",  # 可选: "A100", "T4", "A10G"
    image=image,
    cpu=4,
    memory=1024 * 16,
    timeout=60*10
)
@modal.web_server(port=10088, startup_timeout=600)  # 开启 Web 服务（Flask/FastAPI）
def web():
    os.chdir("/root/flask-server2")
    os.execvp("gunicorn", [
        "gunicorn", "app:app",
        "-b", "0.0.0.0:10088",
        "--workers", "4",
        "--threads", "4",
        "--timeout", "300"
    ])


