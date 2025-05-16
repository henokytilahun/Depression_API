# Dockerfile
FROM python:3.9-slim

# Don’t write pyc files, and stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps for sklearn, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI code + artifacts
COPY main.py .
COPY model.h5 .
COPY scaler.pkl .
COPY selected_features.json .
COPY le_*.pkl .

EXPOSE 8000

# Launch via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
