
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for tokenizers wheels and basic builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
# RUN pip install --upgrade pip \
#  && pip install torch==2.3.0 \
#  && pip install transformers==4.43.3 tokenizers==0.19.1 dash==2.17.1 plotly>=5.22.0 gunicorn>=22.0.0 numpy>=1.24.0
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
#  && pip install --no-cache-dir --no-deps transformers==4.44.2 tokenizers==0.19.1 \
#     dash==2.17.1 plotly>=5.22.0 gunicorn>=22.0.0 numpy>=1.24.0 \
#  && pip check

COPY . /app

# Expose port for fly
ENV PORT=8080
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:server"]
