""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils
import face3d
import vxl
import Pix2Headpose
import Landmarks2Headpose


# Estimate PNCC and Offsets using pix2face network

this_dir = os.path.dirname(__file__)
pix2face_component_dir = os.path.join(this_dir, '../..')


img_fname = os.path.join(this_dir, '../pix2face/data', 'CASIA_0000107_004.jpg')
img = io_utils.imread(img_fname)


# create a list of identical images for the purpose of testing timing
num_test_images = 100
imgs = [img,] * num_test_images

use_pix2face = True  # Set this to False to use sparse landmark-based pose estimation instead.

if use_pix2face:
    # Use dense alignment to estimate pose
    pose_estimator = Pix2Headpose.Pix2Headpose(pix2face_component_dir, cuda_device=0, cuda_v8=False)
else:
    # Use sparse landmarks to estimate pose (slightly faster, but less accurate)
    pose_estimator = Landmarks2Headpose.Landmarks2Headpose(pix2face_component_dir)

import time
t0 = time.time()

# estimate pose for all images in the list
for img in imgs:
   pose = pose_estimator.headpose(img, verbose=True)

t1 = time.time()
total_elapsed = t1 - t0

print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_test_images))
