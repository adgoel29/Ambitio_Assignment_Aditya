FROM python:3.12-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true"]
