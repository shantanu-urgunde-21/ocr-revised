FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_preprocessing.py .
COPY ocr_model.py .
COPY ocr_service.py .
COPY ocr_api.py .
COPY ocr_debug_original.py .

EXPOSE 8000

CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000"]