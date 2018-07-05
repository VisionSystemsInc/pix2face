ARG from=centos:7.2.1511
FROM ${from}
LABEL maintainer="Daniel Crispell <dan@visionsystemsinc.com>"
SHELL ["/usr/bin/env", "bash", "-euxvc"]

RUN yum groupinstall -y "Development Tools"; \
    rm -rf /var/cache/yum/*

RUN yum install -y epel-release; \
    yum install -y \
    ca-certificates which cmake3 \
    openblas-devel \
    python-virtualenv \
    unzip wget \
    python36-devel \
    # OpenGL needed by face3d
    glew-devel glew \
    # face3d builds both dlib and vxl. these both include (different) versions of
    # the libjpeg source and will build it if the libjpeg library is not found,
    # which causes a "multiple definition of" symbol error.
    libjpeg-turbo-devel \
    # keep vxl from building an old version of libtiff from source
    libtiff-devel; \
    if [ "${CUDA_VERSION+set}" != "set" ]; then \
      yum install -y mesa-libEGL-devel mesa-common-dev libosmesa6-dev libglu1-mesa-dev; \
    fi; \
    rm -rf /var/cache/yum/*

# Install python packages in a virtualenv
RUN python36 -m venv /pix2face_venv; \
    # activate doesn't like u flag
    set +u; \
    source /pix2face_venv/bin/activate; \
    pip install --upgrade pip; \
    pip install numpy matplotlib Pillow scikit-image opencv-python-headless; \
    if [ "${CUDA_VERSION+set}" == "set" ]; then \
      pip install http://download.pytorch.org/whl/cu91/torch-0.3.1-cp36-cp36m-linux_x86_64.whl; \
    else \
      pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl; \
    fi; \
    pip install torchvision

ENV GOSU_VERSION 1.10
RUN wget -O /usr/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64";  \
    chmod u+s /usr/bin/gosu; \
    chmod +x /usr/bin/gosu

ARG user_id=1000
RUN useradd -m -s /bin/bash -N -u ${user_id} pix2face_user

USER pix2face_user

WORKDIR /pix2face_super

ENV PYTHONPATH /pix2face_super/pix2face:/pix2face_install:/pix2face_super/python

COPY entrypoint.bsh /entrypoint.bsh
ENTRYPOINT ["/entrypoint.bsh"]

CMD bash
