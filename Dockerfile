# Use a lightweight Python base image
FROM python:3.13-slim-bookworm

# Install Apache and Streamlit dependencies in one go and clean up afterward
RUN apt-get update && \
    apt-get install -y \
    apache2 \
    apache2-utils \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev \
    gcc \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies with no cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the Docker container
COPY app.py /app

# Copy the rest of the app files
COPY . /app/

# Expose necessary port for Streamlit
EXPOSE 2504

# Set up the Apache proxy configurations
RUN echo "ProxyPass /team4s25 http://localhost:2504/team4s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team4s25 http://localhost:2504/team4s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team4s25/(.*) ws://localhost:2504/team4s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable Apache modules for proxy support
RUN a2enmod proxy proxy_http rewrite

# Start Apache and Streamlit using `sh` in the CMD
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.port=2504 --server.baseUrlPath=/team4s25"]
