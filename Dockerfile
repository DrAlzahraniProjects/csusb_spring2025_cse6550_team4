# Stage 1: Build stage
FROM python:3.10-slim as build

# Install build dependencies

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    apache2 \
    apache2-utils \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir scikit-learn

# Set up work directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# Install dependencies (with no cache to reduce size)

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

# Stage 2: Runtime stage (final image)
FROM python:3.10-slim

# Copy Python dependencies from build stage to runtime stage
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /usr/local/bin /usr/local/bin

# Copy only the necessary Apache configuration from the build stage
COPY --from=build /etc/apache2/sites-available/000-default.conf /etc/apache2/sites-available/000-default.conf

# Install minimal runtime dependencies (Apache)
RUN apt-get update && apt-get install --no-install-recommends -y \
    apache2-utils && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages including scikit-learn explicitly
RUN pip install --no-cache-dir \
    streamlit \
    requests \
    pandas \
    langchain-groq \
    langchain \
    python-dotenv \
    scrapy \
    beautifulsoup4 \
    scikit-learn \
    numpy

# Expose port 2504 for Streamlit
EXPOSE 2504

# Start Apache and Streamlit
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.port=2504 --server.baseUrlPath=/team4s25"]
