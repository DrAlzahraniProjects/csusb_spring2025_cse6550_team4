FROM python:3.10-slim

# Copy your app into the container (make sure chatbot.py is in the current directory)
COPY . /app

# Install necessary packages
RUN pip install --upgrade pip
RUN pip install streamlit altair jupyter


# Set the working directory
WORKDIR /app

# Expose ports for both Streamlit and Jupyter
EXPOSE 8501
EXPOSE 8888

# Run the Streamlit app and Jupyter notebook
CMD ["sh", "-c", "streamlit run chatbot.py & jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root"]
