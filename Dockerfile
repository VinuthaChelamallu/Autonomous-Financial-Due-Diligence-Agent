# Dockerfile
# Container for the Financial Due Diligence Agent HTTP service.

FROM python:3.12-slim

# System dependencies (minimal; extend only if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . /app

# Environment variables (set actual values in the cloud runtime)
ENV GOOGLE_API_KEY=""
ENV FINNHUB_API_KEY=""
ENV PORT=8080

# Expose the FastAPI app via Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
