""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import sys
import os
import numpy as np
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils

this_dir = os.path.dirname(__file__)
pix2face_component_dir = os.path.join(this_dir, '../..')

use_pix2face = True  # Set this to False to use sparse landmark-based pose estimation instead.
if use_pix2face:
    import Pix2Headpose
    # Use dense alignment to estimate pose
    # If using cuda 8, set cuda_v8=True for faster runtime.
    pose_estimator = Pix2Headpose.Pix2Headpose(pix2face_component_dir, cuda_device=0, cuda_v8=True)
else:
    import Landmarks2Headpose
    # Use sparse landmarks to estimate pose (slightly faster, but less accurate)
    pose_estimator = Landmarks2Headpose.Landmarks2Headpose(pix2face_component_dir)


def main(input_fname, output_fname):

    with open(output_fname, 'w') as ofd:
        with open(input_fname, 'r') as ifd:
            for line in ifd:
                img_fname = line.strip()
                print(img_fname)
                img = io_utils.imread(img_fname)


                yaw,pitch,roll = pose_estimator.headpose(img, verbose=False)
                yawd, pitchd, rolld = [np.rad2deg(angle) for angle in yaw,pitch,roll]
                ofd.write(img_fname + ' %0.1f %0.1f %0.1f\n' % (yawd,pitchd,rolld))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ' + sys.argv[0] + ' <input_file> <output_file>')
        print('  <input_file> should contain a list of image filenames')
        print('  <output_file> will contain filenames + yaw,pitch,roll values in degrees')
        sys.exit(-1)
    input_fname = sys.argv[1]
    output_fname = sys.argv[2]
    main(input_fname, output_fname)
