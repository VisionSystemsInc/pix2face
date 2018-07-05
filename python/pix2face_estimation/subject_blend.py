import numpy as np
import pix2face.test
from . import mesh_renderer
from . import coefficient_estimation
import face3d
import cv2


class face_blender(object):
    def __init__(self, cuda_device=None, load_pix2face_model=True):
        if load_pix2face_model:
            self.pix2face_net = pix2face.test.load_pretrained_model(cuda_device)
        else:
            self.pix2face_net = None
        self.pix2face_data = coefficient_estimation.load_pix2face_data()
        self.cuda_device = cuda_device
        self.texture_res = 512

    def img2tex(self, img, coeffs):
        renderer = mesh_renderer.get_mesh_renderer()
        head_mesh_warped = face3d.head_mesh(self.pix2face_data.head_mesh)
        head_mesh_warped.apply_coefficients(self.pix2face_data.subject_components,
                                            self.pix2face_data.expression_components,
                                            coeffs.subject_coeffs(), coeffs.expression_coeffs(0))
        tex = face3d.image_to_texture_float(img, head_mesh_warped, coeffs.camera(0), renderer)
        return tex

    def composite_texture(self, img, subj_coeffs, expr_coeffs, camera, tex):
        renderer = mesh_renderer.get_mesh_renderer()
        head_mesh_warped = face3d.head_mesh(self.pix2face_data.head_mesh)
        head_mesh_warped.apply_coefficients(self.pix2face_data.subject_components,
                                            self.pix2face_data.expression_components,
                                            subj_coeffs, expr_coeffs)
        render_walpha = face3d.texture_to_image_float(tex, head_mesh_warped, camera, renderer)
        render_rgb = (render_walpha[:,:,0:3]*255).astype(np.uint8)
        render_mask = (render_walpha[:,:,3] * 255).astype(np.uint8)
        render_mask[render_mask < 10] = 0
        render_mask[render_mask > 0] = 255
        render_mask[0,0] = 255
        render_mask[-1,-1] = 255
        img_out = cv2.seamlessClone(render_rgb, img, render_mask, (img.shape[1]//2, img.shape[0]//2), cv2.NORMAL_CLONE)
        return img_out

    def blend_faces(self, image_list, coeff_list, weights=None):
        weighted_tex_rgb = np.zeros((self.texture_res, self.texture_res, 3), np.float32)
        weight_sum_img = np.zeros((self.texture_res, self.texture_res))
        obs_prob = np.ones((self.texture_res, self.texture_res))
        if weights is None:
            weights = [1.0/len(image_list),] * len(image_list)
        assert len(image_list) == len(coeff_list) == len(weights)
        for img, img_coeffs, weight in zip(image_list, coeff_list, weights):
            tex = self.img2tex(img, img_coeffs)
            tex_rgb = tex[:,:,0:3]/255
            tex_alpha = tex[:,:,3]
            #tex_rgba = np.concatenate((tex_rgb, np.expand_dims(tex_alpha,axis=2)),axis=2)
            weighted_tex_rgb += weight * tex_rgb * np.expand_dims(tex_alpha,2)
            weight_sum_img += weight*tex_alpha
            obs_prob *= (1.0 - tex_alpha)

        obs_prob = 1.0 - obs_prob
        weight_sum_img[weight_sum_img < 1e-6] = 1.0
        weighted_tex_rgb /= np.expand_dims(weight_sum_img,2)
        weighted_tex_rgba = np.concatenate((weighted_tex_rgb, np.expand_dims(obs_prob,2)),axis=2)
        subj_coeffs = np.mean([coeffs.subject_coeffs() for coeffs in coeff_list], axis=0)
        expr_coeffs = np.mean([coeffs.expression_coeffs(0) for coeffs in coeff_list], axis=0)
        final_img = self.composite_texture(image_list[0], subj_coeffs, expr_coeffs, coeff_list[0].camera(0), weighted_tex_rgba)
        return final_img

    def blend_face_images(self, img1, img2, weight1=0.5):
        """
        Blend the face images of img1 and img2 using an estimated 3D model and texture map
        """
        coeffs1 = coefficient_estimation.estimate_coefficients(img1, self.pix2face_net, self.pix2face_data, self.cuda_device)
        coeffs2 = coefficient_estimation.estimate_coefficients(img2, self.pix2face_net, self.pix2face_data, self.cuda_device)
        final_img = self.blend_faces([img1, img2], [coeffs1, coeffs2], [weight1, 1.0-weight1])
        return final_img
