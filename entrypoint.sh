#!/bin/bash
# Start Jupyter Notebook in the background
jupyter notebook --ip=0.0.0.0 --port=2514 --no-browser --allow-root &

# Start Streamlit
streamlit run app.py --server.port=2504 --server.address=0.0.0.0 &

# Wait for both background processes to finish
wait
