""" This Script Demonstrates the basic image -> sparse landmarks --> camera estimation pipeline
"""

import numpy as np
import os
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils
import face3d
import vxl
import dlib


this_dir = os.path.dirname(__file__)
pix2face_data_dir = os.path.join(this_dir, '../pix2face/data/')
dlib_model_dir = os.path.join(this_dir, '../dlib_models/')
dlib_face_detector_model_fname = os.path.join(dlib_model_dir, 'combined_face_detector.svm')
dlib_aflw_landmark_model_fname = os.path.join(dlib_model_dir, 'sp-aflw-depth6-cascade20_nu0.1-frac1.00-splits60.dat')


face_detector = dlib.fhog_object_detector(dlib_face_detector_model_fname)
landmark_detector = dlib.shape_predictor(dlib_aflw_landmark_model_fname)

# The order of the dlib landmarks are different than the canonical AFLW order.
# This array maps from aflw -> dlib indices
aflw_to_dlib_map = [0,1,12,14,15,16,17,18,19,20,2,3,4,5,6,7,8,9,10,11,13]

img_fname = os.path.join(pix2face_data_dir, 'CASIA_0000107_004.jpg')
img = io_utils.imread(img_fname)

num_images = 100
imgs = [img,] * num_images

import time
t0 = time.time()

for img in imgs:
    # detect faces in the image
    face_dets = face_detector(img)
    if len(face_dets) == 0:
        print('No face detections. Skipping')
        continue
    # if multiple faces detected, just take the first
    lms = landmark_detector(img, face_dets[0])
    if (lms.num_parts != 21):
        print('Unexpected Number of landmarks returned from dlib. Skipping.')
        continue

    # convert the landmarks to type vgl_point_2d, in the standard AFLW order
    lms_aflw = [lms.part(aflw_to_dlib_map[i]) for i in range(21)]
    lms_vxl = [vxl.vgl_point_2d(lm.x, lm.y) for lm in lms_aflw]

    # estimate orthographic camera parameters
    cam_params = face3d.compute_camera_params_from_aflw_landmarks(lms_vxl, img.shape[1], img.shape[0])

    # Print Yaw, Pitch, Roll of Head
    R_cam = np.array(cam_params.rotation.as_matrix())  # rotation matrix of estimated camera
    R0 = np.diag((1,-1,-1))  # R0 is the rotation matrix of a frontal camera
    R_head = np.dot(R0,R_cam)
    yaw, pitch, roll = geometry_utils.matrix_to_Euler_angles(R_head, order='YXZ')
    print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))


t1 = time.time()
total_elapsed = t1 - t0
print('Total Elapsed = %0.1f s : Average %0.2f s / image' % (total_elapsed, total_elapsed / num_images))
