# Pix2Face Docker
Instructions for running the pix2face pipeline from within the docker.

### Get docker-compose
download [docker-compose](https://github.com/docker/compose/releases/download/1.21.2/docker-compose-Linux-x86_64) if you don't already have it, and place it in your `PATH`.


### Clone the source code with submodules

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
To build the gpu version, add the `--gpu` flag.


### Run within the docker

Note: If you want to run on the GPU, you'll need nvidia-docker.  Then change the docker image name from "pix2face" to "pix2face-gpu" within the docker scripts.
That is, replace

```bash
docker-compose run pix2face
```

with

```bash
docker-compose run pix2face-gpu
```

The example scripts accept a `--gpu` flag to do this for you.

Rendering (i.e. anything that uses face3d.mesh_renderer) can in principle work without the GPU, but currently fails to create a valid OpenGL context.  Use the gpu version for now if you need to render images.

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
