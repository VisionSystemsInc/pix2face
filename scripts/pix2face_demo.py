""" This Script Demonstrates the basic image -> PNCC + offsets --> coefficient estimation --> 3D Jitter pipeline. """

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

model_fname = os.path.join(pix2face_data_dir, 'models/pix2face_unet_cuda75.pt')
model = pix2face.test.load_model(model_fname)


img_fname = os.path.join(pix2face_data_dir, 'CASIA_0000107_004.jpg')
img = io_utils.imread(img_fname)
print('Estimating PNCC + Offsets..')
outputs = pix2face.test.test(model, [img,])
pncc = outputs[0][0]
offsets = outputs[0][1]
print('..Done')


pvr_data_dir = os.path.join(this_dir, '../janus/components/pvr/data_3DMM/')
debug_dir = ''
debug_mode = False


# load needed data files
head_mesh = face3d.head_mesh(pvr_data_dir)
subject_components = np.load(os.path.join(pvr_data_dir, 'pca_components_subject.npy'))
expression_components = np.load(os.path.join(pvr_data_dir, 'pca_components_expression.npy'))
subject_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_subject.npy'))
expression_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_expression.npy'))
renderer = face3d.mesh_renderer()


# create coefficient estimator
coeff_estimator = face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh, subject_components, expression_components, subject_ranges, expression_ranges, debug_mode, debug_dir)


# Estimate Coefficients from PNCC and Offsets
print('Estimating Coefficients..')
img_ids = ['img0',]
coeffs = coeff_estimator.estimate_coefficients_perspective(img_ids, [pncc,], [offsets,])
print('..Done.')

# Print Yaw, Pitch, Roll of Head
R_cam = np.array(coeffs.camera(0).rotation.as_matrix()) # rotation matrix of estimated camera
R0 = np.diag((1,-1,-1))  # R0 is the rotation matrix of a frontal camera
R_head = np.dot(R0,R_cam)
yaw, pitch, roll = geometry_utils.matrix_to_Euler_angles(R_head, order='YXZ')
print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))

# Render 3D-Jittered Images
print('Rendering Jittered Images..')
jitterer = face3d.media_jitterer_perspective([img,], coeffs, head_mesh, subject_components, expression_components, renderer, "")

# manually alter expression
new_expression_coeffs = np.zeros_like(coeffs.expression_coeffs(0))
new_expression_coeffs[1] = 2.0
new_expression_coeffs[2] = -2.0
new_expression_coeffs[14] = 1.5
render_expr = jitterer.render_perspective(coeffs.camera(0), coeffs.subject_coeffs(), new_expression_coeffs, subject_components, expression_components)


# manually alter pose
delta_R = vxl.vgl_rotation_3d(geometry_utils.Euler_angles_to_quaternion(np.pi/3, 0, 0, order='YXZ'))
cam = coeffs.camera(0)
new_R = cam.rotation * delta_R
new_cam = face3d.perspective_camera_parameters(cam.focal_len, cam.principal_point, new_R, cam.translation, cam.nx, cam.ny)
render_rot = jitterer.render_perspective(new_cam, coeffs.subject_coeffs(), coeffs.expression_coeffs(0), subject_components, expression_components)
print('..Done.')

# save out results
output_dir = this_dir
print('Saving results')
io_utils.imwrite(img, os.path.join(output_dir, "image_original.jpg"))
io_utils.imwrite(render_expr, os.path.join(output_dir, "image_expression_jitter.jpg"))
io_utils.imwrite(render_rot, os.path.join(output_dir, "image_pose_jitter.jpg"))
