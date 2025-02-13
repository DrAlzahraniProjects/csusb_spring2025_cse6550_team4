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
