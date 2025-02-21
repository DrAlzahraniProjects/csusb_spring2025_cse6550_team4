#!/bin/bash
# Start Jupyter Notebook in the background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

# Start Streamlit
streamlit run chatbot.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for both background processes to finish
wait
