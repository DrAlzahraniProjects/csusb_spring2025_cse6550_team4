FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code into the container
COPY . /app

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the ports used by Jupyter and Streamlit
EXPOSE 8888 8501

# Start the application via the entrypoint script
CMD ["./entrypoint.sh"]
