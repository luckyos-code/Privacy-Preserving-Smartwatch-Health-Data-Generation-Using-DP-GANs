FROM tensorflow/tensorflow:2.12.0-gpu

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev libcairo2-dev pkg-config python3.8-dev libcairo2 ffmpeg libsm6 libxext6

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

COPY requirements-docker.txt requirements.txt
RUN pip3.8 install --upgrade pip wheel setuptools --no-cache-dir && \
    pip3.8 install -r requirements.txt --no-cache-dir

WORKDIR /app
