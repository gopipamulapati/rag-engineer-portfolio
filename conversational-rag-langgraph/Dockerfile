FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MEMORY_BACKEND=sqlite SQLITE_PATH=/data/sessions.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt langchain-openai

COPY . .
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
