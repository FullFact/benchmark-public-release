# Score explorer (src/explorer/app.py) for Cloud Run. Installs only what the
# explorer needs, not the full benchmark dependencies (torch etc.).
# See src/explorer/DEPLOY.md.
FROM python:3.12-slim
WORKDIR /app
COPY src/explorer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/benchmark-2026-07-13-to-2026-09-22.sqlite data/atom_offsets.csv data/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
CMD exec uvicorn explorer.app:app --host 0.0.0.0 --port ${PORT:-8080}
