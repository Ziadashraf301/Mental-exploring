FROM apache/airflow:2.10.2

# Switch to root to install system packages FIRST
USER root

RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # General build tools for Python packages
    gcc \
    g++ \
    build-essential \
    # Git (needed for some pip installations)
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements
COPY airflow/docker/requirements.txt /tmp/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"