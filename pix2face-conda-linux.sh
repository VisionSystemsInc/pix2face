# Source this file to set up pix2face environment.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# add anaconda to path
export PATH="$DIR"/../anaconda2/bin:$PATH
export JANUS_ROOT="$DIR"/janus_root
# source the janus script
source "$DIR"/janus/scripts/janus-conda-linux.sh
# add pix2face to pythonpath
export PYTHONPATH="$DIR"/pix2face:$PYTHONPATH

