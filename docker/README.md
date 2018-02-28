# Pix2Face Docker
Instructions for running the pix2face pipeline from within the docker.

### Clone the source code

If you didn't use the --recursive flag when cloning, you'll need to run:
```bash
git submodule update --init --recursive
```


### Download the datafiles

```bash
./download_data.bsh
```


### Build the docker image

```bash
./docker/docker-build.bsh
```


### Build the software within the docker

```bash
./docker/build.bsh
```


### Run within the docker

#### Pose Estimation
```bash
./docker/run_pose_estimation.bsh <image_dir> <output_dir>
```
A single csv file (poses.csv) will be written to `<output_dir>` containing yaw, pitch, and roll for each image in `<image_dir>`

#### Coefficient Estimation
```bash
./docker/run_coeff_estimation.bsh <image_dir> <output_dir>
```
One coefficients file per image in `<image_dir>` will be written to `<output_dir>`

