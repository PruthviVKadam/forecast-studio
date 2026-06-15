# Serving image for the Dash app. Bundled data/ caches make it run offline.
FROM python:3.12-slim

# libgomp1 -> XGBoost OpenMP runtime; build-essential helps any source builds (prophet/cmdstanpy).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY assets/ assets/
COPY data/ data/
COPY app.py .

EXPOSE 7860
ENV PORT=7860
# app:server is the Flask WSGI app exposed by Dash.
CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 180"]
