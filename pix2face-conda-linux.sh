# Source this file to set up pix2face environment.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# add anaconda to path
export JANUS_CONDA_ROOT="$DIR"/../anaconda2
export PATH="$JANUS_CONDA_ROOT"/bin:$PATH
#export JANUS_ROOT="$DIR"/janus_root
export JANUS_PLATFORM=centos7
export JANUS_RELEASE=${JANUS_ROOT}/${JANUS_PLATFORM}  # symlinked to nfs mount
export NVIDIA_LIB_DIR=/usr/lib64/nvidia
export LD_LIBRARY_PATH=$JANUS_CONDA_ROOT/lib:$NVIDIA_LIB_DIR:$LD_LIBRARY_PATH  # for PVR binaries

# source the janus script
# add pix2face to pythonpath
export JANUS_SOURCE="$DIR"/janus
export PYTHONPATH=${JANUS_SOURCE}/python:${PYTHONPATH}  # prepend with janus source
export PYTHONPATH="$DIR"/pix2face:$PYTHONPATH

