FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# libgomp is needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: download data if missing, train, then serve the dashboard.
#   docker build -t smd-lab .
#   docker run -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" smd-lab
EXPOSE 8000
CMD ["sh", "-c", "[ -f runs/LATEST ] || python -m src.pipeline all; python -m dashboard.server --host 0.0.0.0 --port 8000"]
