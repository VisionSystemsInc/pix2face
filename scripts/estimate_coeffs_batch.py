""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import sys
import os
import numpy as np
from PIL import Image
import pix2face.test
import pix2face_estimation.coefficient_estimation

cuda_device = 0
num_subject_coeffs=30
num_expression_coeffs=20

this_dir = os.path.dirname(__file__)
pvr_data_dir = os.path.join(this_dir,'../janus/components/pvr/data_3DMM')

def main(input_fname, output_dir):
    # load the 3DMM data
    mm_data = pix2face_estimation.coefficient_estimation.load_pix2face_data(pvr_data_dir, num_subject_coeffs, num_expression_coeffs)
    # load the pix2face network
    pix2face_net = pix2face.test.load_pretrained_model(cuda_device=cuda_device)
    # open input and output files, process one file per line
    with open(input_fname, 'r') as ifd:
        for line in ifd:
            img_fname = line.strip()
            print(img_fname)
            img = np.array(Image.open(img_fname))
            try:
                coeffs = pix2face_estimation.coefficient_estimation.estimate_coefficients(img, pix2face_net, mm_data, cuda_device=cuda_device)
            except pix2face_estimation.coefficient_estimation.CoefficientEstimationError:
                print('Failed to estimate coefficients for ' + img_fname)
                continue

            basename = os.path.splitext(os.path.basename(img_fname))[0]
            output_fname = os.path.join(output_dir, basename + '_coeffs.txt')
            coeffs.save(output_fname)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ' + sys.argv[0] + ' <input_file> <output_dir>')
        print('  <input_file> should contain a list of image filenames')
        print('  One coefficients file per image will be written to <output_dir>.')
        sys.exit(-1)
    input_fname = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_fname, output_dir)
