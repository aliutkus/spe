FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y \
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

WORKDIR /spe
COPY . /spe

RUN python -m pip install --upgrade pip
RUN git submodule init && git submodule update
RUN cd experiments/lra && pip install -e ./fast_attention ./long-range-arena ../../src/jax

ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone git@github.com:maximzubkov/positional-bias.git
RUN cd positional-bias && pip install -e .

RUN pip install --upgrade jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install -r experiments/lra/requirements.txt