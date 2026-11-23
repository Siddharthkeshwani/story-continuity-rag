FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && pip install -r requirements.txt
COPY --chown=user:user . .
RUN chown user:user /home/user/app
USER user
EXPOSE 7860
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.port=7860","--server.headless=true"]

