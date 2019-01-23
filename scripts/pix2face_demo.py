""" This Script Demonstrates the basic image -> PNCC + offsets --> coefficient estimation --> 3D Jitter pipeline. """

import numpy as np
import os
import pix2face_estimation.geometry_utils as geometry_utils
import face3d
import vxl
import pix2face
from PIL import Image


# Estimate PNCC and Offsets using pix2face network

cuda_device = 0

this_dir = os.path.dirname(__file__)
pix2face_data_dir = os.path.join(this_dir, '../pix2face_net/data/')

model = pix2face.test.load_pretrained_model(cuda_device=cuda_device)

img_fname = os.path.join(pix2face_data_dir, 'CASIA_0000107_004.jpg')
img = np.array(Image.open(img_fname))
print('Estimating PNCC + Offsets..')
outputs = pix2face.test.test(model, [img,], cuda_device=cuda_device)
pncc = outputs[0][0]
offsets = outputs[0][1]
print('..Done')


pvr_data_dir = os.path.join(this_dir, '../face3d/data_3DMM/')
debug_dir = ''
debug_mode = False

num_subject_coeffs = 199  # max 199
num_expression_coeffs = 29  # max 29

# load needed data files
head_mesh = face3d.head_mesh(pvr_data_dir)
subject_components = np.load(os.path.join(pvr_data_dir, 'pca_components_subject.npy'))
expression_components = np.load(os.path.join(pvr_data_dir, 'pca_components_expression.npy'))
subject_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_subject.npy'))
expression_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_expression.npy'))

# keep only the PCA components that we will be estimating
subject_components = vxl.vnl.matrix(subject_components[0:num_subject_coeffs,:])
expression_components = vxl.vnl.matrix(expression_components[0:num_expression_coeffs,:])
subject_ranges = vxl.vnl.matrix(subject_ranges[0:num_subject_coeffs,:])
expression_ranges = vxl.vnl.matrix(expression_ranges[0:num_expression_coeffs,:])

# create rendering object (encapsulates OpenGL context)
renderer = face3d.mesh_renderer()
# create coefficient estimator
coeff_estimator = face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh, subject_components, expression_components, subject_ranges, expression_ranges, debug_mode, debug_dir)

# Estimate Coefficients from PNCC and Offsets
print('Estimating Coefficients..')
img_ids = ['img0',]
coeffs, result = coeff_estimator.estimate_coefficients_perspective(img_ids, [pncc,], [offsets,])
if not result.success:
    raise Exception('ERROR estimating coefficents for ' + img_fname)
print('..Done.')

# Print Yaw, Pitch, Roll of Head
R_cam = np.array(coeffs.camera(0).rotation.as_matrix())  # rotation matrix of estimated camera
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
render_expr = jitterer.render(coeffs.camera(0), coeffs.subject_coeffs(), new_expression_coeffs, subject_components, expression_components)


# manually alter pose
delta_R = vxl.vgl.rotation_3d(geometry_utils.Euler_angles_to_quaternion(np.pi/3, 0, 0, order='YXZ'))
cam = coeffs.camera(0)
new_R = cam.rotation * delta_R
new_cam = face3d.perspective_camera_parameters(cam.focal_len, cam.principal_point, new_R, cam.translation, cam.nx, cam.ny)
render_rot = jitterer.render(new_cam, coeffs.subject_coeffs(), coeffs.expression_coeffs(0), subject_components, expression_components)
print('..Done.')

# save out results
output_dir = this_dir
print('Saving results')
Image.fromarray(img).save(os.path.join(output_dir, "image_original.jpg"))
Image.fromarray(render_expr[:,:,0:3]).save(os.path.join(output_dir, "image_expression_jitter.jpg"))
Image.fromarray(render_rot[:,:,0:3]).save(os.path.join(output_dir, "image_pose_jitter.jpg"))
