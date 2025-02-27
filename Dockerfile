FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for Faiss
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code into the container
COPY . /app

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the ports used by Streamlit and Jupyter
EXPOSE 2504 2514

# Start the application via the entrypoint script
CMD ["./entrypoint.sh"]
