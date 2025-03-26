# Use a lightweight Python base image
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies for Streamlit
RUN apt-get update && \
    apt-get install -y \
    libxml2-dev \
    libxslt-dev \
    gcc \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies with no cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the Docker container
COPY app.py /app

# Copy the rest of the app files
COPY . /app/

# Expose necessary port for Streamlit
EXPOSE 2594

# Start Streamlit using the new port
CMD ["streamlit", "run", "app.py", "--server.port=2594", "--server.baseUrlPath=/team4s25"]