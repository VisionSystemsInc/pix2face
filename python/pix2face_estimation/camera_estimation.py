import face3d
import pix2face.test
from .camera_decomposition import decompose_camera_rotation


def estimate_camera(image, pix2face_net, cuda_device=0):
    PNCC, offsets = pix2face.test.test(pix2face_net, image, cuda_device=cuda_device)
    camera_params = face3d.compute_camera_params_from_pncc_and_offsets_perspective(PNCC, offsets)
    return camera_params


def extract_head_pose(camera_params):
    yaw, pitch, roll = decompose_camera_rotation(camera_params.rotation.as_matrix(), pitch_offset=-7)
    return yaw, pitch, roll


def estimate_head_pose(image, pix2face_net, cuda_device=0):
    return extract_head_pose(estimate_camera(image, pix2face_net, cuda_device=cuda_device))
