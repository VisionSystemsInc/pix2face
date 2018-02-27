FROM centos:7.2.1511
LABEL maintainer="Daniel Crispell <dan@visionsystemsinc.com>"

RUN yum groupinstall -y "Development Tools" && \
    rm -rf /var/cache/yum/*

RUN yum install -y epel-release && \
    yum install -y \
    ca-certificates which cmake3 \
    # needed by numpy
    # includes a version of lapack; don't need the devel; NOTE not openblas-threads64
    openblas-threads \
    python-virtualenv unzip \
    # includes a version of lapack; don't need the devel
    python-devel python-pip openblas-threads \
    # OpenGL needed by janus:pvr
    mesa-libEGL-devel.x86_64 mesa-libGL-devel.x86_64 glew-devel glew \
    #libXrandr-devel libXinerama-devel glfw-devel libXcursor-devel libXi-devel \
    # janus builds both dlib and vxl. these both include (different) versions of
    # the libjpeg source and will build it if the libjpeg library is not found,
    # which causes a 'multiple definition of' symbol error.
    libjpeg-turbo-devel \
    # keep vxl from building an old version of libtiff from source
    libtiff-devel \
    && \
    rm -rf /var/cache/yum/*

# Install python packages
RUN pip install --upgrade pip && \
    pip install numpy matplotlib Pillow scikit-image

# Install pytorch
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl && \
    pip install torchvision

ARG user_id=1000
RUN useradd -m -s /bin/bash -N -u ${user_id} pix2face_user

USER pix2face_user

WORKDIR /src/pix2face_super

ENV PYTHONPATH /pix2face_super/pix2face:/pix2face_super/janus/python:/pix2face_super/docker/build/janus/lib

CMD python --version
