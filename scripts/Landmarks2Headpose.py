import numpy as np
import matplotlib.pyplot as plt
import os
import janus.pvr.python_util.io_utils as io_utils
import janus.pvr.python_util.geometry_utils as geometry_utils
import face3d
import vxl
import dlib

# The order of the dlib landmarks are different than the canonical AFLW order.
# This array maps from aflw -> dlib indices
aflw_to_dlib_map = [0,1,12,14,15,16,17,18,19,20,2,3,4,5,6,7,8,9,10,11,13]

class Landmarks2Headpose(object):
    """Basic class that implements head pose estimation from sparse 2D AFLW landmarks """
    def __init__(self, pix2facedir):
        """Initialize the pix2face object by preloading models.  pix2facedir is an absolute path to the pix2face/ component directory
        """

        dlib_model_dir = os.path.join(pix2facedir, 'pix2face_super/dlib_models/')
        dlib_face_detector_model_fname = os.path.join(dlib_model_dir, 'combined_face_detector.svm')
        dlib_aflw_landmark_model_fname = os.path.join(dlib_model_dir, 'sp-aflw-depth6-cascade20_nu0.1-frac1.00-splits60.dat')

        self.face_detector = dlib.fhog_object_detector(dlib_face_detector_model_fname)
        self.landmark_detector = dlib.shape_predictor(dlib_aflw_landmark_model_fname)

        # add 7 degrees to pitch value so that pitch=0 -> "Frankfurt Horizontal Plane" is horizontal.
        self.pitch_offset = np.deg2rad(-7.0)


    def headpose(self, img, verbose=False):
        """
        Accept a numpy array image of a cropped face as input.
        Returns a list of (yaw, pitch, roll) tuples in radians of the face in the input image.
        """
        # detect faces in the image
        face_dets = self.face_detector(img)
        if len(face_dets) == 0:
            if verbose:
                print('No faces detected - using entire image.')
            face_dets = [dlib.rectangle(0,0,img.shape[1],img.shape[0]),]

        # if multiple faces detected, just take the first
        lms = self.landmark_detector(img, face_dets[0])
        if (lms.num_parts != 21):
            raise Exception('Unexpected Number of landmarks returned from dlib')

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
        pitch += self.pitch_offset

        if verbose:
            print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))

        return (yaw, pitch, roll)

