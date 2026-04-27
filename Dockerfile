FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
COPY . .
EXPOSE 8501
ENV FORCE_CPU=true
ENV TORCH_DEVICE=cpu
CMD ["streamlit", "run", "Inference.py"]