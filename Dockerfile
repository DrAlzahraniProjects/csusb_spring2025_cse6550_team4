# Stage 1: Build stage
FROM python:3.10-slim as build

# Install build dependencies (minimal required ones)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    apache2 \
    apache2-utils \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev && \
    apt-get clean


# Set up work directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy your Python code into the Docker container
COPY . /app

# Expose port for Streamlit
EXPOSE 2504

# Set up the Apache proxy configurations
RUN echo "ProxyPass /team4s25 http://localhost:2504/team4s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team4s25 http://localhost:2504/team4s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team4s25/(.*) ws://localhost:2504/team4s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable Apache modules for proxy and WebSocket support
RUN a2enmod proxy proxy_http rewrite

# Start Apache and Streamlit using `sh` in the CMD
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.port=2504 --server.baseUrlPath=/team4s25"]
