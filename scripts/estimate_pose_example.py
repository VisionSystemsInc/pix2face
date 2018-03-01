""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import numpy as np
import os
from PIL import Image
import pix2face.test
import pix2face_estimation.camera_estimation

# Set this to an integer value to run on a CUDA device, None for CPU.
cuda_device = None

this_dir = os.path.dirname(__file__)
img_fname = os.path.join(this_dir, '../pix2face/data', 'CASIA_0000107_004.jpg')
img = np.array(Image.open(img_fname))

# create a list of identical images for the purpose of testing timing
num_test_images = 100
imgs = [img,] * num_test_images


# Use dense alignment to estimate pose
pix2face_net = pix2face.test.load_pretrained_model(cuda_device)

import time
t0 = time.time()

# estimate pose for all images in the list
for img in imgs:
   pose = pix2face_estimation.camera_estimation.estimate_head_pose(img, pix2face_net, cuda_device)

t1 = time.time()
total_elapsed = t1 - t0

print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f' % pose)
print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_test_images))
