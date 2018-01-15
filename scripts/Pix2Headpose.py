import numpy as np
import os
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils
import face3d
import vxl
import pix2face

class Pix2Headpose(object):
    """Basic Pix2Face class that implements head pose estimation from dense 3D alignment"""
    def __init__(self, pix2facedir, cuda_device=None, cuda_v8 = False):
        """Initialize the pix2face object by preloading models.  pix2facedir is an absolute path to the pix2face/ component directory
        """

        #this_dir = os.path.dirname(__file__)
        pix2face_data_dir = os.path.join(pix2facedir, 'pix2face_super', 'pix2face', 'data/')

        model_fname = os.path.join(pix2face_data_dir, 'models/pix2face_unet_v10.pt')

        self.model = pix2face.test.load_model(model_fname)
        self.cuda_device = cuda_device

        # add 7 degrees to pitch value so that pitch=0 -> "Frankfurt Horizontal Plane" is horizontal.
        self.pitch_offset = np.deg2rad(-7.0)

    def headpose(self, img, verbose=False):
        """
        Accept a numpy array image of a cropped face as input.
        Returns a list of (yaw, pitch, roll) tuples in radians of the face in the input image.
        """
        # Estimate PNCC and Offsets using pix2face network
        if verbose:
            print('Estimating PNCC + Offsets..')
        pncc, offsets = pix2face.test.test(self.model, img, cuda_device=self.cuda_device)
        if verbose:
            print('..Done')

        cam_params = face3d.compute_camera_params_from_pncc_and_offsets_ortho(pncc, offsets)

        # Extract Yaw, Pitch, Roll of Head
        R_cam = np.array(cam_params.rotation.as_matrix())  # rotation matrix of estimated camera
        R0 = np.diag((1,-1,-1))  # R0 is the rotation matrix of a frontal camera
        R_head = np.dot(R0,R_cam)
        yaw, pitch, roll = geometry_utils.matrix_to_Euler_angles(R_head, order='YXZ')
        pitch += self.pitch_offset

        if verbose:
            print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))

        return (yaw, pitch, roll)

