# Match the Python 3.13 version used during training/serialization
FROM python:3.13-slim

ENV GIT_PYTHON_REFRESH=quiet

# Install LightGBM OpenMP dependency and clean apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies first (better Docker layer caching)
COPY requirements-dockers.txt ./
RUN pip install --no-cache-dir -r requirements-dockers.txt

# Copy application files and required directories
COPY app.py ./
COPY run_information.json ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean.py ./scripts/data_clean.py
COPY ./static ./static

# Document the port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]