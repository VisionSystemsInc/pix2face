FROM nvidia/cudagl:9.2-devel-ubuntu18.04
ARG cpu_image=0
ENV CUDNN_VERSION 7.4.1.5
SHELL ["/usr/bin/env", "bash", "-euxvc"]
RUN apt-get update && apt-get install -y --no-install-recommends\
            libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 && \
        apt-mark hold libcudnn7
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git unzip python3-dev cmake ninja-build libz-dev libssl-dev libboost-dev libglew-dev libtiff-dev python3-venv libjpeg-turbo8-dev \
        openssh-client libopenblas-dev; \
        rm -rf /var/lib/apt/lists/*

# Python env
ADD requirements.txt /pix2face/requirements.txt
ENV CPU_ONLY=${cpu_image}
RUN python3 -m venv /pix2face/venv;\
    # activate doesn't like u flag
    set +u; \
    source /pix2face/venv/bin/activate; \
    pip install --upgrade pip; \
    pip install -r /pix2face/requirements.txt; \
    if [ $CPU_ONLY -eq 0 ]; then \
      pip install torch; \
    else \
      pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl; \
    fi; \
    pip install torchvision

ENV GOSU_VERSION 1.10
RUN wget -O /usr/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64";  \
    chmod u+s /usr/bin/gosu; \
    chmod +x /usr/bin/gosu

COPY entrypoint.bsh /entrypoint.bsh
RUN gosu root chmod 755 /entrypoint.bsh
ENTRYPOINT ["/entrypoint.bsh"]
CMD bash
