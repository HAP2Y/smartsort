# Slim base image used by the rules-worker (filename rules only, no LLM,
# no text extraction). Ships ~150 MB lighter than the AI image because it
# skips PyMuPDF (native), python-docx, and their transitive build deps.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./

# Core deps + redis client. No build-essential needed; everything in the
# base set is pure-Python wheels.
RUN pip install --no-cache-dir \
        "typer>=0.12.3" \
        "rich>=13.7.1" \
        "PyYAML>=6.0.1" \
        "requests>=2.31.0" \
        "redis>=5.0.0"

COPY classifier ./classifier
COPY inference ./inference
COPY movers ./movers
COPY config ./config
COPY main.py ./main.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
