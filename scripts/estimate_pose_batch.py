""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import sys
import numpy as np
from PIL import Image, ImageFile
import pix2face.test
import pix2face_estimation.camera_estimation
import os
# Set this to an integer to run on a CUDA device, None to run on the CPU.
cpu_only = int(os.environ.get("CPU_ONLY")) != 0
cuda_device = None if cpu_only else 0
if cpu_only:
    print("Running on CPU")
else:
    print("Running on cuda device %s" % cuda_device)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(input_fname, output_fname):
    # load the pix2face network
    pix2face_net = pix2face.test.load_pretrained_model(cuda_device=cuda_device)
    # open input and output files, process one file per line
    with open(output_fname, 'w') as ofd, open(input_fname, 'r') as ifd:
        # write header for output
        ofd.write('FILENAME, HEAD_YAW, HEAD_PITCH, HEAD_ROLL\n')
        # for each line in the input file
        for line in ifd:
            img_fname = line.strip()
            print(img_fname)
            # load the image
            img = np.array(Image.open(img_fname))
            # estimate yaw,pitch,roll
            pose = pix2face_estimation.camera_estimation.estimate_head_pose(img, pix2face_net, cuda_device=cuda_device)
            # write out the pose values to the output CSV file
            ofd.write(img_fname + ', %0.1f, %0.1f, %0.1f\n' % pose)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ' + sys.argv[0] + ' <input_file> <output_file>')
        print('  <input_file> should contain a list of image filenames')
        print('  <output_file> will contain lines of the form filename,yaw,pitch,roll. Pose angels have units degrees')
        sys.exit(-1)
    input_fname = sys.argv[1]
    output_fname = sys.argv[2]
    main(input_fname, output_fname)
