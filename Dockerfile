FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
