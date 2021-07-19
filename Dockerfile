FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
                       build-essential \
                       ca-certificates \
                       wget \
                       unzip \
                       ssh \
                       cmake \
                       git \
                       vim \
                       python3-dev python3-pip python3-setuptools

RUN ln -sf $(which python3) /usr/bin/python \
    && ln -sf $(which pip3) /usr/bin/pip

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /long-range-arena
COPY . /long-range-arena

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone git@github.com:maximzubkov/positional-bias.git
RUN cd positional-bias && pip install -e .