FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/FIU-Neuro/brainconn#egg=brainconn

COPY . .