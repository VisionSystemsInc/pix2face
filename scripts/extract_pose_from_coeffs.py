import os
import sys
import face3d
import pix2face_estimation.camera_estimation as camera_estimation


def main(coeffs_dir, output_csv_fname):
    """
    Read all coeffs files from coeffs_dir, and save all poses in a single csv file.
    """
    assert os.path.isdir(coeffs_dir), "coeffs_dir not a directory"
    coeff_fnames = sorted(os.listdir(coeffs_dir))

    num_images = 0
    with open(output_csv_fname, 'w') as fd:
        fd.write("FILENAME,YAW,PITCH,ROLL\n")
        for coeff_fname in coeff_fnames:
            coeff_path = os.path.join(coeffs_dir, coeff_fname)
            coeffs = face3d.subject_perspective_sighting_coefficients(coeff_path)
            for i in range(coeffs.num_sightings):
                yaw, pitch, roll = camera_estimation.extract_head_pose(coeffs.camera(i))
                chip_fname = coeffs.image_filename(i)
                fd.write("{}, {:0.3f}, {:0.3f}, {:0.3f}\n".format(chip_fname, yaw, pitch, roll))
                num_images += 1

    print("wrote {} poses from {} files".format(num_images, len(coeff_fnames)))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: {} <coeffs_dir> <output_csv_fname>".format(sys.argv[0]))
        sys.exit(-1)
    coeffs_dir = sys.argv[1]
    output_csv_fname = sys.argv[2]
    main(coeffs_dir, output_csv_fname)
