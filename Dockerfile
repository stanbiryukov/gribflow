FROM condaforge/mambaforge:4.9.2-5 as build
ENV DEBIAN_FRONTEND=noninteractive

# lib requirements
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libssl-dev \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY environment.yaml .
RUN mamba env create -f environment.yaml
# Install conda-pack:
RUN mamba install -c conda-forge conda-pack==0.6.0
# Use conda-pack to create a standalone env in /venv:
RUN conda-pack -n python3 -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

# Distro
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=build /venv /venv
# lib requirements
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3-opencv \
    libeccodes-dev \
    software-properties-common \
    build-essential \
    libssl-dev \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# switch from /bin/sh to /bin/bash
SHELL ["/bin/bash", "-c"]
# ENTRYPOINT source /venv/bin/activate
# make sure env is activated on entry
RUN echo -e "\
source /venv/bin/activate \n\
" >> ~/.bashrc

# install gribflow
WORKDIR /app
COPY . /app
RUN /venv/bin/python3 /app/setup.py install
