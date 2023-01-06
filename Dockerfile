FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN set -xe \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install software-properties-common curl -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install python3.7 python3.7-distutils python3.7-dev -y \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.7

RUN apt update && apt install ffmpeg libsm6 libxext6 gcc -y

# RUN curl -sSL https://sdk.cloud.google.com | bash

# ENV PATH $PATH:/root/google-cloud-sdk/bin

WORKDIR /workspace/
ADD . / /workspace/
# ADD requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt
# RUN pip install -r requirements/dev.txt
ENV PYTHONPATH "${PYTHONPATH}:/workspace"