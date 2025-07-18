FROM python:3.10-slim
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ api/
COPY templates/ templates/
COPY static/ static/
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://localhost:6379
CMD redis-server --daemonize yes && uvicorn api.app:app --host 0.0.0.0 --port $PORT
