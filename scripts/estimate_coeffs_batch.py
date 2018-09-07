""" This Script Demonstrates the basic image -> PNCC + offsets --> camera estimation pipeline
"""
import sys
import os
import numpy as np
from PIL import Image, ImageFile
import pix2face.test
import pix2face_estimation.coefficient_estimation
from torch.multiprocessing import Pool

# set cuda_device to an integer value to run on a GPU, set to None to run on CPU
cuda_device = 0

num_subject_coeffs=30
num_expression_coeffs=20

#num_dirs_in_id = 3  #  /a/b/c.jpg -> a_b_c_coeffs.txt
num_dirs_in_id = 1  # /a/b/c.jpg -> c_coeffs.txt

num_threads = 4

ImageFile.LOAD_TRUNCATED_IMAGES = True

this_dir = os.path.dirname(__file__)
pvr_data_dir = os.path.join(this_dir,'../face3d/data_3DMM')

# load the 3DMM data
mm_data = pix2face_estimation.coefficient_estimation.load_pix2face_data(pvr_data_dir, num_subject_coeffs, num_expression_coeffs)


def process_chunk(fnames):
    # load the pix2face network
    pix2face_net = pix2face.test.load_pretrained_model(cuda_device=cuda_device)
    images = [np.array(Image.open(f)) for f in fnames]

    coeffs_list = pix2face_estimation.coefficient_estimation.estimate_coefficients_batch(images, pix2face_net, mm_data, cuda_device=cuda_device, img_labels=fnames)
    assert len(coeffs_list) == len(fnames)
    for coeffs, img_fname in zip(coeffs_list, fnames):
        print(img_fname)
        if coeffs is None:
            print('Failed to estimate coefficients for ' + img_fname)
            continue
        splitpath = os.path.normpath(img_fname).split(os.sep)
        basename = os.path.splitext('_'.join(splitpath[-num_dirs_in_id:]))[0]
        output_fname = os.path.join(output_dir, basename + '_coeffs.txt')
        coeffs.save(output_fname)
    return


def main(input_fname, output_dir):
    # open input file, read one filename per line
    img_fnames = []
    with open(input_fname, 'r') as ifd:
        for line in ifd:
            img_fname = line.strip()
            # handle CSVs where the first column is the filename.
            # This will obviously break if the filenames contain commas.
            img_fname = img_fname.split(',')[0]
            img_fnames.append(img_fname)
    print('Read %d image filenames' % len(img_fnames))

    chunk_size = 128
    chunks = [img_fnames[i:i+chunk_size] for i in range(0,len(img_fnames),chunk_size)]
    if num_threads > 1:
        pool = Pool(num_threads)
        _ = pool.map(process_chunk, chunks)
    else:
        for chunk in chunks:
            process_chunk(chunk)
    print('done.')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ' + sys.argv[0] + ' <input_file> <output_dir>')
        print('  <input_file> should contain a list of image filenames')
        print('  One coefficients file per image will be written to <output_dir>.')
        sys.exit(-1)
    input_fname = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_fname, output_dir)
