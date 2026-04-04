FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .
RUN groupadd -r app && useradd -r -g app app
USER app
VOLUME ["/data"]
CMD ["python", "-m", "matrix_llm_bot", "--config", "/data/config.json"]
