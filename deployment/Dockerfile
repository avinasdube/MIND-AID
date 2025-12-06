# Placeholder Dockerfile for MIND-AID demo services
FROM python:3.11-slim
WORKDIR /app

# Install baseline requirements first; later phases can layer on
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (adjust as phases mature)
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8501
CMD ["python", "src/app.py"]
