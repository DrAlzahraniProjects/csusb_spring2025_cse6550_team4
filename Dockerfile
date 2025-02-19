FROM python:3.10-slim

# Install necessary packages
RUN pip install --upgrade pip && pip install streamlit altair jupyter

# Set a working directory inside the container
WORKDIR /app

# Copy your app into the container
COPY . /app

# Expose both Jupyter and Streamlit ports
EXPOSE 8501 8888

# Start both Jupyter and Streamlit correctly
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & streamlit run chatbot.py --server.port=8501 --server.address=0.0.0.0
FROM python:3.10-slim

# Install necessary packages
RUN pip install --upgrade pip && pip install streamlit altair jupyter

# Set a working directory inside the container
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose both Jupyter and Streamlit ports
EXPOSE 8501 8888

# Start both services using the entrypoint script
CMD ["./entrypoint.sh"]
