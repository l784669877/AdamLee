# 基础镜像
FROM python:3.11.8-slim
# 设置⼯作⽬录
WORKDIR /app
# 升级 pip
RUN pip install --upgrade pip
# 复制依赖⽂件到⼯作⽬录
COPY requirements.txt /app/requirements.txt
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt
# 复制项⽬⽂件到⼯作⽬录
COPY . /app
# 暴露端⼝
EXPOSE 8000
# 启动服务
CMD ["python", "app.py"]