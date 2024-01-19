FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt update && \
    apt install -y \
        python3 \
        python3-dev \
        python3-pip \
        build-essential \
        wget \
        curl \
        git \
        vim && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY install.sh /tmp/install.sh
RUN chmod +x /tmp/install.sh && \
    bash /tmp/install.sh && \
    rm /tmp/install.sh

WORKDIR /workspaces/jax_pallas_testing
