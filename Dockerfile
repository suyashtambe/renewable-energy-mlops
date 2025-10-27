# --------------------------------------------------------
# 1Base image
# --------------------------------------------------------
FROM python:3.10-slim

# ====== Working Directory ======
WORKDIR /app

# ====== Copy Everything ======
COPY . /app

# ====== Install System Dependencies ======
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ====== Install Python Requirements ======
RUN pip install --no-cache-dir -r requirements.txt

# ====== Ensure Models Exist ======
# If models are not built into the image, create an empty directory
RUN mkdir -p models

# ====== Set Environment Variables ======
ENV PYTHONUNBUFFERED=1

# ====== Run API ======
CMD ["python", "src/api.py"]
