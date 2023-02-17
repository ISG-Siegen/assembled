FROM python:3.8.10

RUN apt-get update && apt-get install build-essential swig python-dev -y && \
	pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
