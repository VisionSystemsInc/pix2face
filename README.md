# pix2face_super
## "Superproject" wrapping components of the complete pix2face pipeline, including:
   * The pix2face dense face alignment and 3D estimation network
   * The camera/pose and face coefficient estimation contained in the Janus repository
![](pix2face_super_teaser.png "pix2face_teaser")

## Requirements
You have two options for running: within a docker container, or natively on your system. The only requirement for the former is docker itself.  The requirements for building and running natively are:
   * CMake (version 2.8.12 or higher)
   * Python (versions 2.7 and 3.5 have been tested)
   * numpy, Pillow, scikit-image
   * pytorch (version 0.3.0.post4 has been tested)

Optional:
   * CUDA (for GPU acceleration of pix2face network and coefficient estimation)

For a full list of system packages required, consult the [Dockerfile](./docker/Dockerfile).

## Setup and Running (Docker)
See the [README.md](./docker/README.md) in the docker directory of this repository.


## Setup (Native)
Clone this repository to `$PIX2FACE_SRC_DIR`. If you did not clone recursively, you'll need to run:
```bash
cd $PIX2FACE_SRC_DIR
git submodule update --init --recursive
```

Create a build directory and run Cmake:
```bash
mkdir $PIX2FACE_BUILD_DIR
cd $PIX2FACE_BUILD_DIR
cmake $PIX2FACE_SRC_DIR
```

Build the Janus binaries:
```bash
make install
```

Configure your python path:

```bash
export PYTHONPATH=$PIX2FACE_SRC_DIR/pix2face:$PIX2FACE_SRC_DIR/janus/python:$PIX2FACE_BUILD_DIR/janus/lib
```

Download the required data files to their correct locations by running the `download_data.bsh` script:
```bash
cd $PIX2FACE_SRC_DIR
./download_data.bsh
```


## Running (Native)
The scripts directory contains several examples of pose and coefficient estimation. For example, to generate a CSV file `$POSE_CSV_FNAME` with yaw, pitch, and roll (in degrees) of every face image listed (one per line) in the text file `$IMAGE_PATHS_FILE`:
``` bash
python scripts/estimate_pose_batch.py $IMAGE_PATHS_FILE $POSE_CSV_FNAME
```


## Citation
If you find this software useful, please consider referencing:

```bibtex
@INPROCEEDINGS{pix2face2017,
author = {Daniel Crispell and Maxim Bazik},
booktitle = {2017 IEEE International Conference on Computer Vision Workshop (ICCVW)},
title = {Pix2Face: Direct 3D Face Model Estimation},
year = {2017},
pages = {2512-2518},
ISSN = {2473-9944},
month={Oct.}
}
```


## Contact
Daniel Crispell [dan@visionsystemsinc.com](mailto:dan@visionsystemsinc.com)
