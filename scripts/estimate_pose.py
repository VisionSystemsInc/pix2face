""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils
import face3d
import vxl
import pix2face


# Estimate PNCC and Offsets using pix2face network

this_dir = os.path.dirname(__file__)
pix2face_data_dir = os.path.join(this_dir, '../pix2face/data/')

model_fname = os.path.join(pix2face_data_dir, 'models/pix2face_unet_cuda80.pt')
model = pix2face.test.load_model(model_fname)


img_fname = os.path.join(pix2face_data_dir, 'CASIA_0000107_004.jpg')
img = io_utils.imread(img_fname)

num_images = 100
imgs = [img,] * num_images

import time
t0 = time.time()

print('Estimating PNCC + Offsets..')
outputs = pix2face.test.test(model, imgs, cuda_device=0)
print('..Done')
for pncc, offsets in outputs:
    #pncc = outputs[0][0]
    #offsets = outputs[0][1]

    cam_params = face3d.compute_camera_params_from_pncc_and_offsets_ortho(pncc, offsets)

    # Print Yaw, Pitch, Roll of Head
    R_cam = np.array(cam_params.rotation.as_matrix())  # rotation matrix of estimated camera
    R0 = np.diag((1,-1,-1))  # R0 is the rotation matrix of a frontal camera
    R_head = np.dot(R0,R_cam)
    yaw, pitch, roll = geometry_utils.matrix_to_Euler_angles(R_head, order='YXZ')
    print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))
t1 = time.time()
total_elapsed = t1 - t0
print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_images))
