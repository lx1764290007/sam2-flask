module.exports = {
    apps: [
      {
        name: "flask-sam2",      // 应用名称
        script: "bash", // 指定虚拟环境中的 gunicorn 解释器
        args: "app:app",          // Flask 应用（`app.py` 文件中的 Flask 实例）
        interpreter: "/bin/bash",  // 使用 bash 来执行命令
        interpreter_args: "-c 'source /usr/local/segment_flask/server/myenv/bin/activate && cd /usr/local/segment_flask/server && gunicorn -w 4 -b 0.0.0.0:7263 app:app'",  // 激活虚拟环境并启动 gunicorn
        env: {
          "FLASK_ENV": "production", // 设置环境变量（如果需要）
        },
        watch: true,  // 启用文件监听，自动重启
      },
    ],
  };
  