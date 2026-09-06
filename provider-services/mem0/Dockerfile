FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ARG MEM0_SOURCE_ARCHIVE_SHA256=5443d9dd99196e33fdde31ef663518c022a8f4a86ef032e7b4bd7b00285705c2
ARG MEM0_BUILD_INPUT_SHA256

LABEL org.opencontainers.image.title="OAMB Mem0 REST API" \
      org.opencontainers.image.source="https://github.com/mem0ai/mem0" \
      org.opencontainers.image.version="2.0.19" \
      org.opencontainers.image.revision="dc82354e143c2581d505d581a00286d6ef8c3605" \
      io.oamb.mem0.source-archive-sha256="${MEM0_SOURCE_ARCHIVE_SHA256}" \
      io.oamb.mem0.build-input-sha256="${MEM0_BUILD_INPUT_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=8

COPY mem0/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

COPY .build/mem0-source.tar /tmp/mem0-source.tar
RUN python -c "import hashlib,pathlib; p=pathlib.Path('/tmp/mem0-source.tar'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '${MEM0_SOURCE_ARCHIVE_SHA256}'"
ADD .build/mem0-source.tar /opt/
COPY mem0/extraction_response.py /opt/mem0-source/mem0/memory/oamb_extraction_response.py
RUN python /opt/mem0-source/mem0/memory/oamb_extraction_response.py /opt/mem0-source/mem0/memory/main.py \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/mem0-source \
    && python -c "import importlib.metadata as m; assert m.version('mem0ai') == '2.0.19'"

WORKDIR /app
RUN cp -a /opt/mem0-source/server/. /app/
COPY mem0/inspector.py /opt/oamb/inspector.py

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
