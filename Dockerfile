FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

ENV APP_HOME /app
WORKDIR ${APP_HOME}

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

COPY . .

CMD python3 app.py