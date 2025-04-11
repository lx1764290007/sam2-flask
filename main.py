import modal

image = (
    modal.Image.debian_slim()

    .apt_install("git", "libgl1")  # OpenCV 需要这个库
    .run_commands(
        "git clone https://github.com/lx1764290007/sam2-flask.git /root/sam2"
    )
    .pip_install("gunicorn")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("checkpoints/sam2.1_hiera_base_plus.pt", "/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                    copy=True)  # 模型
)
app = modal.App(name="sam2-flask-app", image=image)


@app.function(
    gpu="A10G",  # 可选: "A100", "T4", "A10G"
    image=image,
    cpu=4,
    memory=1024 * 16,
    timeout=600
)
@modal.web_server(port=10086)  # 开启 Web 服务（Flask/FastAPI）
def web():
    import subprocess
    subprocess.run([
        "gunicorn", "app:app",
        "-b", "0.0.0.0:10086",
        "--workers", "2",
        "--threads", "4",
        "--timeout", "300"
    ], cwd="/root/sam2")

