FROM nvcr.io/nvidia/tensorflow:19.07-py3

USER root
ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV DEBIAN_FRONTEND noninteractive
ARG nproc=2

WORKDIR /tmp

RUN apt-get update && \
    apt-get dist-upgrade -y
    

