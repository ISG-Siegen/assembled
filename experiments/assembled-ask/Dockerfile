FROM python:3.8.10

RUN apt-get update && apt-get install build-essential swig python-dev -y && \
	pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir auto-sklearn

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
