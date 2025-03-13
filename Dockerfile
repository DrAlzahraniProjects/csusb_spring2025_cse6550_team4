FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for Faiss and Apache
RUN apt-get update && apt-get install -y \
    apache2 \
    apache2-utils \
    libopenblas-dev \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code into the container
COPY . /app

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set up the Apache proxy configurations for team4s25
RUN echo "ProxyPass \"/team4s25/jupyter\" \"http://localhost:2514/team4s25/jupyter\"" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse \"/team4s25/jupyter\" \"http://localhost:2514/team4s25/jupyter\"" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPass \"/team4s25\" \"http://localhost:2504/team4s25\"" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse \"/team4s25\" \"http://localhost:2504/team4s25\"" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team4s25/(.*) ws://localhost:2504/team4s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable Apache modules for proxy, WebSocket support, and rewriting
RUN a2enmod proxy proxy_http proxy_uwsgi rewrite

# Expose the ports for Streamlit and Jupyter
EXPOSE 2504 2514

# Start the application via the entrypoint script (starts Apache, Streamlit, and Jupyter)
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.maxUploadSize=10 --server.port=2504 --server.baseUrlPath=/team4s25 & jupyter notebook --port=2514 --ip=0.0.0.0 --NotebookApp.base_url=/team4s25/jupyter --NotebookApp.notebook_dir=/app --NotebookApp.token='' --allow-root"]
