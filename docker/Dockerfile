FROM centos:7.2.1511
LABEL maintainer="Daniel Crispell <dan@visionsystemsinc.com>"

RUN yum groupinstall -y "Development Tools" && \
    rm -rf /var/cache/yum/*

RUN yum install -y epel-release && \
    yum install -y \
    ca-certificates which cmake3 \
    openblas-devel \
    python-virtualenv \
    unzip \
    python36-devel \
    # OpenGL needed by face3d
    mesa-libEGL-devel.x86_64 mesa-libGL-devel.x86_64 glew-devel glew \
    glew-devel glew \
    gdb \
    #libXrandr-devel libXinerama-devel glfw-devel libXcursor-devel libXi-devel \
    # janus builds both dlib and vxl. these both include (different) versions of
    # the libjpeg source and will build it if the libjpeg library is not found,
    # which causes a 'multiple definition of' symbol error.
    libjpeg-turbo-devel \
    # keep vxl from building an old version of libtiff from source
    libtiff-devel \
    && \
    rm -rf /var/cache/yum/*

# Install python packages in a virtualenv
RUN python36 -m venv /pix2face_venv
RUN source /pix2face_venv/bin/activate && \
    pip install --upgrade pip && \
    pip install numpy matplotlib Pillow scikit-image && \
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  && \
    pip install torchvision

ARG user_id=1000
RUN useradd -m -s /bin/bash -N -u ${user_id} pix2face_user

USER pix2face_user

WORKDIR /pix2face_super

ENV PYTHONPATH /pix2face_super/pix2face:/pix2face_super/docker/install

COPY entrypoint.bsh /entrypoint.bsh
ENTRYPOINT ["/entrypoint.bsh"]

CMD bash
