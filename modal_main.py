import modal
import sys
sys.path.append("/root/flask-server")  # 👈 把 server 目录加入 Python 路径

image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "libgl1",  # OpenCV 图形相关
        "libglib2.0-0",  # OpenCV 多线程相关（libgthread 所属包）
    )
    .run_commands(
        "git clone https://github.com/lx1764290007/sam2-flask.git /root/flask-server"
    )
    .pip_install("gunicorn")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("checkpoints/sam2.1_hiera_base_plus.pt", "/root/flask-server/checkpoints/sam2.1_hiera_base_plus.pt",
                    copy=True)  # 模型
    .add_local_dir("sam2/configs", "/root/flask-server/configs")
)
app = modal.App(name="sam-web-app2", image=image)


@app.function(
    gpu="T4",  # 可选: "A100", "T4", "A10G"
    image=image,
    cpu=4,
    memory=1024 * 16,
    timeout=60*10
)
@modal.web_server(port=10088)  # 开启 Web 服务（Flask/FastAPI）
def web():
    from app import app as flask_app  # 注意这里的 app 是 Flask 实例
    return flask_app


