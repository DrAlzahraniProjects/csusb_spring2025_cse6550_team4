# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install necessary dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Expose ports for both Streamlit and Jupyter
EXPOSE 8501
EXPOSE 8888

# Command to run both Streamlit and Jupyter
CMD ["sh", "-c", "streamlit run chatbot.py --server.port 8501 & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
