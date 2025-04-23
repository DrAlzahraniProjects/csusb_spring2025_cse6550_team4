# Use a lightweight Python base image
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install system libs
RUN apt-get update && \
    apt-get install -y \
      libxml2-dev \
      libxslt-dev \
      libffi-dev \
      build-essential \
      gcc \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ─────────────── pytorch for cpu only ───────────────
# Install CPU‑only PyTorch + sentence-transformers from the PyTorch index
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch sentence-transformers
# ─────────────────────────────────────────────

# Install the rest of your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the container
COPY app.py /app

# Copy the rest of the app files
COPY . /app/

# Expose necessary port for Streamlit
EXPOSE 2504

# Start Streamlit using the new port
CMD ["streamlit", "run", "app.py", "--server.port=2504", "--server.baseUrlPath=/team4s25"]
