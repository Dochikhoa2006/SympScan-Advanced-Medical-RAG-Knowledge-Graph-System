FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y default-jdk build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader wordnet omw-1.4
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
CMD ["streamlit", "run", "app.py"]