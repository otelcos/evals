
FROM python:3.13-slim
# Starts with a base Python 3.10 slim image

# Updates package list and installs necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    python3-tk \
    zip \
    unzip \
    procps \
    r-base \
    gdal-bin \
    libgdal-dev

RUN pip install --upgrade pip

COPY docker-requirements.txt /tmp/docker-requirements.txt
# Install packages using the copied requirements file
RUN pip install --timeout 1200 -r /tmp/docker-requirements.txt
