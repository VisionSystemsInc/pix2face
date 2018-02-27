""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import sys
import os
import numpy as np
from PIL import Image
import pix2face.test
import pix2face_estimation.camera_estimation

cuda_device = None

def main(input_fname, output_fname):
    # load the pix2face network
    pix2face_net = pix2face.test.load_pretrained_model(cuda_device=cuda_device)
    # open input and output files, process one file per line
    with open(output_fname, 'w') as ofd, open(input_fname, 'r') as ifd:
        # write header for output
        ofd.write('FILENAME, HEAD_YAW, HEAD_PITCH, HEAD_ROLL\n')
        for line in ifd:
            img_fname = line.strip()
            print(img_fname)
            img = np.array(Image.open(img_fname))
            pose = pix2face_estimation.camera_estimation.estimate_head_pose(img, pix2face_net, cuda_device=cuda_device)
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
