FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p chroma_db static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]