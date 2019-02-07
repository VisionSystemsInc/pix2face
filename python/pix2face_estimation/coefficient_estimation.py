"""
3-D Morphable Model (3DMM) coefficient estimation using the pix2face network
"""
import os
from collections import namedtuple
import numpy as np
import vxl
import face3d
import pix2face.test
from . import mesh_renderer


Pix2FaceData = namedtuple('Pix2FaceData',['head_mesh','subject_components','expression_components','subject_ranges','expression_ranges', 'coeff_estimator', 'use_offsets'])


def load_pix2face_data(pvr_data_dir=None, num_subject_coeffs=None, num_expression_coeffs=None, use_offsets_for_estimation=True):
    """
    Load PCA components and ranges.
    Returns a structure containing the following  matrices loaded as instances of vxl.vnl_matrix
    """
    if pvr_data_dir is None:
        # guess the pvr_data_dir
        this_dir = os.path.dirname(__file__)
        pvr_data_dir = os.path.join(this_dir, '../../face3d/data_3DMM')

    # load data files as numpy arrays
    head_mesh = face3d.head_mesh(pvr_data_dir)
    subject_components = np.load(os.path.join(pvr_data_dir, 'pca_components_subject.npy'))
    expression_components = np.load(os.path.join(pvr_data_dir, 'pca_components_expression.npy'))
    subject_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_subject.npy'))
    expression_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_expression.npy'))

    if num_subject_coeffs is None:
        num_subject_coeffs = subject_components.shape[0]
    if num_expression_coeffs is None:
        num_expression_coeffs = expression_components.shape[0]

    # keep only needed rows of subject and expression matrices and convert to vnl matrices
    subject_components = vxl.vnl.matrix(subject_components[0:num_subject_coeffs,:])
    subject_ranges = vxl.vnl.matrix(subject_ranges[0:num_subject_coeffs,:])
    expression_components = vxl.vnl.matrix(expression_components[0:num_expression_coeffs,:])
    expression_ranges = vxl.vnl.matrix(expression_ranges[0:num_expression_coeffs,:])

    debug_mode = False
    debug_dir = ""

    if use_offsets_for_estimation:
        coeff_estimator = \
            face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh,
                                                                    subject_components, expression_components,
                                                                    subject_ranges, expression_ranges,
                                                                    debug_mode, debug_dir)
    else:
        coeff_estimator = \
            face3d.media_coefficient_from_PNCC_estimator(head_mesh,
                                                         subject_components, expression_components,
                                                         subject_ranges, expression_ranges,
                                                         debug_mode, debug_dir)

    # return in MMData structure
    return Pix2FaceData(head_mesh=head_mesh,
                        subject_components=subject_components, subject_ranges=subject_ranges,
                        expression_components=expression_components, expression_ranges=expression_ranges,
                        coeff_estimator=coeff_estimator,
                        use_offsets=use_offsets_for_estimation)


class CoefficientEstimationError(RuntimeError):
    """ Error returned if coefficient estimation fails """
    pass


def estimate_coefficients(image, pix2face_net, pix2face_data, cuda_device=0, img_label=None):
    """
    Estimate shape, expression, and camera parameters for a single image
    """
    PNCC, offsets = pix2face.test.test(pix2face_net, image, cuda_device=cuda_device)

    if cuda_device is not None:
        print("Setting face3d cuda_device to", cuda_device)
        face3d.set_cuda_device(cuda_device)
    if img_label is None:
        img_label = 'img0'

    est_args = [[PNCC,],]
    if pix2face_data.use_offsets:
        est_args.append([offsets,])
    coeffs, result = pix2face_data.coeff_estimator.estimate_coefficients_perspective([img_label], *est_args)
    if not result.success:
        raise CoefficientEstimationError("Coefficient Estimation Failed")
    return coeffs


def estimate_coefficients_joint(images, pix2face_net, pix2face_data, cuda_device=0, img_labels=None):
    """
    Estimate coefficients for multiple images.  A single set of subject coefficients
    will be estimated, and results returned in a single object.
    """
    results = pix2face.test.test(pix2face_net, images, cuda_device=cuda_device)
    PNCCs, offsets = zip(*results)
    if img_labels is None:
        img_labels = ['img%d' % i for i in range(len(images))]
    assert len(img_labels) == len(images)

    if cuda_device is not None:
        face3d.set_cuda_device(cuda_device)

    est_args = [PNCCs,]
    if pix2face_data.use_offsets:
        est_args.append(offsets,)
    coeffs, result = pix2face_data.coeff_estimator.estimate_coefficients_perspective(img_labels, *est_args)
    if not result.success:
        raise CoefficientEstimationError("Coefficient Estimation Failed")
    return coeffs


def estimate_coefficients_batch(images, pix2face_net, pix2face_data, cuda_device=0, img_labels=None):
    """
    Estimate coefficients for multiple images, independently.
    One coeffs object per image will be returned.  If coeff estimation fails, None will be inserted into the list of coefficients.
    """
    results = pix2face.test.test(pix2face_net, images, cuda_device=cuda_device)
    coeffs_list = []

    if img_labels is None:
        img_labels = ['img%d' % i for i in range(len(images))]
    assert len(img_labels) == len(images)

    if cuda_device is not None:
        face3d.set_cuda_device(cuda_device)

    for (PNCC, offsets), img_label in zip(results, img_labels):

        est_args = [[PNCC,],]
        if pix2face_data.use_offsets:
            est_args.append([offsets,])
        coeffs, result = pix2face_data.coeff_estimator.estimate_coefficients_perspective([img_label,], *est_args)
        if not result.success:
            coeffs_list.append(None)
        else:
            coeffs_list.append(coeffs)
    return coeffs_list


def render_coefficients(coeffs, pix2face_data, img_idx=0):
    """
    Returns an image of the face described by coeffs
    pix2face_data should be created by calling load_pix2face_data
    """
    # lazily create renderer (of which there should only be one.)
    renderer = mesh_renderer.get_mesh_renderer()

    texture_res = 64  # resolution not important as this will be a solid color texture
    green_tex = np.zeros((texture_res,texture_res,3), np.uint8)
    green_tex[:,:,1] = 255
    head_mesh_warped = face3d.head_mesh(pix2face_data.head_mesh)
    head_mesh_warped.apply_coefficients(pix2face_data.subject_components,
                                        pix2face_data.expression_components,
                                        coeffs.subject_coeffs(),
                                        coeffs.expression_coeffs(img_idx))
    meshes = head_mesh_warped.meshes()
    for mesh in meshes:
        mesh.set_texture(green_tex)
    renderer.set_ambient_weight(0.5)
    synth = renderer.render(meshes, coeffs.camera(img_idx))
    return synth
