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
RUN cd experiments/lra && pip install ./fast_attention ./long-range-arena ../../src/jax

ARG GIT_TOKEN
RUN git config --global url."https://${GIT_TOKEN}:@github.com/".insteadOf "https://github.com/"

RUN git clone https://github.com/maximzubkov/positional-bias.git
RUN cd positional-bias && pip install .

RUN pip install --upgrade jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install -r experiments/lra/requirements.txt