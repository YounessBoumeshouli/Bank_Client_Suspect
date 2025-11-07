# Partir d'une image Python officielle
FROM python:3.8-slim AS base
RUN apt-get update && \
    apt-get install -y default-jre-headless && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
COPY wait-for-spark.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/wait-for-spark.sh
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["wait-for-spark.sh","streamlit", "run", "main.py"]
