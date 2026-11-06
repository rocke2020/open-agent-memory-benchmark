FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/oam
COPY . /opt/oam
RUN pip install --no-cache-dir uv==0.9.28 && uv sync --locked --no-dev && useradd --create-home --uid 10001 oam
ENV PATH="/opt/oam/.venv/bin:$PATH"
USER oam
EXPOSE 8080
ENTRYPOINT ["oam"]
CMD ["api", "--host", "0.0.0.0", "--port", "8080"]
