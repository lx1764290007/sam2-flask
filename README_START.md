# 直接通过gunicorn启动flask
```bash
cd /usr/local/segment_flask/server 
source myenv/bin/activate 
```
```bash
gunicorn -w 8 -b 0.0.0.0:7263 --timeout 120 app:app # -w：进程数 --timeout 超时（秒）
```
# 通过pm2启动
```bash
cd /usr/local/segment_flask 
pm2 start start.config.js
```
