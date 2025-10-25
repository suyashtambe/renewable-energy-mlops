# # ====== Base Image ======
# FROM python:3.10-slim

# # ====== Working Directory ======
# WORKDIR /app

# # ====== Copy Everything ======
# COPY . /app

# # ====== Install System Dependencies ======
# RUN apt-get update && apt-get install -y \
#     gcc \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # ====== Install Python Requirements ======
# RUN pip install --no-cache-dir -r requirements.txt

# # ====== Ensure Models Exist ======
# # If models are not built into the image, create an empty directory
# RUN mkdir -p models

# # ====== Set Environment Variables ======
# ENV PYTHONUNBUFFERED=1

# # ====== Run API ======
# CMD ["python", "src/api.py"]

# ===============================
# Renewable Energy MLOps Dockerfile
# ===============================
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Pull data from DVC remote
RUN dvc pull -v || true

# Default command to train model
CMD ["python", "src/model_train.py"]
