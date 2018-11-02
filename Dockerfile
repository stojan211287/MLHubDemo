FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y python3 \
                       python3-dev \
                       curl && \
    apt-get clean

COPY requirements.txt /

RUN curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install -r requirements.txt

RUN mkdir -p /code
COPY ./*.py /code/