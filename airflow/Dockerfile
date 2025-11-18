FROM apache/airflow:2.10.2

# Switch to root to install system packages FIRST
USER root

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements
COPY airflow/requirements.txt /tmp/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt