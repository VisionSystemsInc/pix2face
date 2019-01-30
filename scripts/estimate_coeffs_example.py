""" This Script Demonstrates the basic image -> PNCC + offsets --> coefficient estimation pipeline
"""
import numpy as np
import os
from PIL import Image
import pix2face.test
import pix2face_estimation.coefficient_estimation

# Set this to an integer value to run on a CUDA device, None for CPU.
cpu_only = int(os.environ.get("CPU_ONLY")) != 0
cuda_device = None if cpu_only else 0
if cpu_only:
    print("Running on CPU")
else:
    print("Running on cuda device %s" % cuda_device)


this_dir = os.path.dirname(__file__)
img_fname = os.path.join(this_dir, '../pix2face_net/data', 'CASIA_0000107_004.jpg')
img = np.array(Image.open(img_fname))

# create a list of identical images for the purpose of testing timing
num_test_images = 10
imgs = [img,] * num_test_images

# Use dense alignment to estimate pose
pix2face_net = pix2face.test.load_pretrained_model(cuda_device)
pix2face_data = pix2face_estimation.coefficient_estimation.load_pix2face_data(num_subject_coeffs=30, num_expression_coeffs=20)

import time
t0 = time.time()

# estimate pose for all images in the list
for img in imgs:
    coeffs = pix2face_estimation.coefficient_estimation.estimate_coefficients(img, pix2face_net, pix2face_data, cuda_device)

t1 = time.time()
total_elapsed = t1 - t0

print('coeffs = ' + str(coeffs))
print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_test_images))
