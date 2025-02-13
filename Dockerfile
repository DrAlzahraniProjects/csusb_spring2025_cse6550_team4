# Example of Dockerfile
FROM python:3.10-slim

# Install necessary packages
RUN pip install --upgrade pip
RUN pip install streamlit altair

# Copy your app into the container (make sure chatbot.py is in the current directory)
COPY . /app

# Set the working directory
WORKDIR /app

# Expose port
EXPOSE 8501

# Run the Streamlit app (update to use chatbot.py)
CMD ["streamlit", "run", "chatbot.py"]

