FROM python:3.11-slim-bookworm

ARG UV_VERSION=0.11.19
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==${UV_VERSION}

WORKDIR /opt/mu-course
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --extra cpu --extra plots --no-editable
ENV PATH="/opt/mu-course/.venv/bin:$PATH" \
    SPARK_HOME="/opt/mu-course/.venv/lib/python3.11/site-packages/pyspark" \
    PYSPARK_PYTHON="/opt/mu-course/.venv/bin/python" \
    PYSPARK_DRIVER_PYTHON="/opt/mu-course/.venv/bin/python" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY scripts/environment_smoke.py scripts/environment_smoke.py
COPY scripts/submit.py scripts/submit.py
COPY scripts/predict_submit.py scripts/predict_submit.py
COPY scripts/plot_example_data.py scripts/plot_example_data.py
RUN useradd --create-home --uid 10001 course \
    && mkdir -p /workspace/data /workspace/outputs /tmp/spark-work \
    && chown -R course:course /workspace /tmp/spark-work
USER course
CMD ["python", "scripts/environment_smoke.py", "--backend", "cpu"]
