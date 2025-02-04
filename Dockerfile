# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install \
    jupyter \
    streamlit

# Copy all files
COPY . .

# Expose ports for Streamlit (8501) and Jupyter (8888)
EXPOSE 8501 8888

# Set entrypoint for overriding
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]