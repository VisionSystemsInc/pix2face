# Pix2Face Docker
Instructions for running the pix2face pipeline from within the docker.

### Get docker-compose
download [docker-compose](https://github.com/docker/compose/releases/download/1.21.2/docker-compose-Linux-x86_64) if you don't already have it, and place it in your `PATH`.


### Clone the source code with submodules

If you didn't use the --recursive flag when cloning, you'll need to run:
```bash
git submodule update --init --recursive
```


### Build the docker image
Default build is with GPU support. Use the ```--cpu-only``` flag to disable it. This flag is supported by all scripts below.

```bash
./docker/build_docker_image.bsh [--cpu-only]
```


### Build the software within the docker. This script will also download all required data files.

```bash
./docker/build_pix2face_sources.bsh [--cpu-only]
```


### Run bash inside a named pix2face container


```bash
./docker/run_pix2face_interactively.bsh [--cpu-only] [INSTANCE_NAME]
```

Rendering (i.e. anything that uses face3d.mesh_renderer) can in principle work without the GPU, but currently fails to create a valid OpenGL context.  Use the gpu version for now if you need to render images.

#### Examples: Pose and Coefficient Estimation
After a succseful build these two examples should work
```bash
./docker/run_pose_estimation_example.bsh [--cpu-only]
./docker/run_coeff_estimation_example.bsh [--cpu-only]
```

#### Pose Estimation
```bash
./docker/run_pose_estimation.bsh <image_dir> <output_dir> [--cpu-only]
```
A single csv file (poses.csv) will be written to `<output_dir>` containing yaw, pitch, and roll for each image in `<image_dir>`

#### Coefficient Estimation
```bash
./docker/run_coeff_estimation.bsh <image_dir> <output_dir> [--cpu-only]
```
One coefficients file per image in `<image_dir>` will be written to `<output_dir>`

#### Jupyter/IPython Notebook
You can run the pix2face demo notebook on port 8885 of your machine with the command:
```bash
./docker/run_pix2face_notebook.bsh [--cpu-only]
```
