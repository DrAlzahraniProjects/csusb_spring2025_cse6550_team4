# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose ports for Streamlit (8501) and Jupyter (8888)
EXPOSE 8501 8888

# Set entrypoint for overriding 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
