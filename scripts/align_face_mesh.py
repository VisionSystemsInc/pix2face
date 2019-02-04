import eos
import numpy as np
import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '..', 'face3d', 'data_3DMM')
    if not os.path.isdir(data_dir):
        print("Could not find directory %s. Did you download the data using download_data.bsh?" % data_dir)
        sys.exit(-1)
    model_path = os.path.join(data_dir, "sfm_shape_3448.bin")
    if not os.path.isfile(model_path):
        print("Could not find model at %s. Did you download the data using download_data.bsh?" % model_path)
        sys.exit(-1)
    model = eos.morphablemodel.load_model(model_path)
    mean_model = model.get_mean()
    shape_model = model.get_shape_model()
    texture_coords = np.array(model.get_texture_coordinates())
    mean_vertices = np.array(mean_model.vertices)
    triangles = np.array(shape_model.get_triangle_list())
    transform_path = os.path.join(script_dir, "basel_to_surrey_rigid_transform.txt")

    b_to_s = np.loadtxt(transform_path)
    s_to_b = np.linalg.inv(b_to_s)

    mean_vertices = np.hstack((mean_vertices, np.ones((mean_vertices.shape[0], 1))))
    mean_vertices = (s_to_b.dot(mean_vertices.T) ).T[:,0:3]

    def fmt_print(string, array):
        return '\n'.join([string.format(*x) for x in array])

    ply_str = '''ply
    format ascii 1.0
    comment surrey face mesh
    element vertex {}
    property float x
    property float y
    property float z
    property float s
    property float t
    element face {}
    property list uchar int vertex_indices
    end_header
    {}
    {}
    '''.format(
        len(mean_vertices),
        len(triangles),
        fmt_print("{} {} {} {} {}", np.hstack((mean_vertices, texture_coords))),
        fmt_print("3 {} {} {}", triangles))
    mean_face_head = ply_str

    # subject
    pca_coeff_stds_subject = np.sqrt(shape_model.get_eigenvalues()).reshape(-1,1)
    pca_coeff_ranges_subject = np.hstack((pca_coeff_stds_subject,pca_coeff_stds_subject)) * np.array((-4,4))
    pca_components_subject = shape_model.get_orthonormal_pca_basis().T
    # expression
    deformations = []
    names = []
    blend_shapes_path = os.path.join(data_dir, "expression_blendshapes_3448.bin")
    if not os.path.isfile(blend_shapes_path):
        print("Could not find the blended shapes at %s. Did you download the data using download_data.bsh?" % blend_shapes_path)
        sys.exit()
    blend = eos.morphablemodel.load_blendshapes(blend_shapes_path)
    for b in blend:
        deformations.append(b.deformation)
        names.append(b.name)

    pca_coeff_stds_expression = np.ones((6,1))
    pca_coeff_ranges_expression = np.ones((6,2)) * np.array((-4,4))
    pca_components_expression = np.array(deformations)
    expression_meanings = " ".join(names)
# save

    x = os.path.join(data_dir,"expression_meanings.txt")
    with open(x, 'w') as f:
        f.write(expression_meanings)

    x = os.path.join(data_dir,"mean_face_head.ply")
    with open(x, 'w') as f:
        f.write(mean_face_head)

    np.save(os.path.join(data_dir, "pca_coeff_ranges_expression.npy"), pca_coeff_ranges_expression)
    np.save(os.path.join(data_dir, "pca_coeff_ranges_subject.npy"), pca_coeff_ranges_subject)
    np.save(os.path.join(data_dir, "pca_coeff_stds_expression.npy"), pca_coeff_stds_expression)
    np.save(os.path.join(data_dir, "pca_coeff_stds_subject.npy"), pca_coeff_stds_subject)
    np.save(os.path.join(data_dir, "pca_components_expression.npy"), pca_components_expression)
    np.save(os.path.join(data_dir, "pca_components_subject.npy"), pca_components_subject)
