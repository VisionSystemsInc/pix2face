import os
import face3d
from janus.openbr import readbinary
import time
import numpy as np
from PIL import Image
from multiprocessing import Pool
import sys
janus_root = os.environ.get('JANUS_ROOT')
janus_data = os.environ.get('JANUS_DATA')
pix2face_root = os.environ.get('PIX2FACE_SOURCE')
pvr_data_path = os.path.join(pix2face_root, 'janus', 'components', 'pvr', 'data_3DMM')
if not os.path.isdir(pvr_data_path):
    print("%s does not exist!" % pvr_data_path)
    sys.exit()
msceleb1m_coeffs_path = '/data3/data/msceleb1m/msceleb1m_filtered_plus_vggface2_coeffs.csv'
msceleb1m_jitter_dir = '/data3/data/msceleb1m/msceleb1m_jitters'
min_yaw = 30; max_yaw = 90
egl_device = 5
force_rewrite = False
N_jitters = 1
jobs = 4;
print_rate = 1000

def category_index(path):
    return os.path.basename(os.path.dirname(path))


def chip_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def process_line(line, jitterer):
    parts = line.split(',')  # expects the chip path and the corresponding coefficient file path separated by comma
    chip_path = parts[0]
    coeff_path = parts[1]
    if not os.path.isfile(coeff_path):
        # print('coeff for  %s does not exist' % chip_path)
        return
    id = category_index(coeff_path)
    # print("ID found %s" % id)
    output_dir = os.path.join(msceleb1m_jitter_dir, id)
    if not os.path.isdir(output_dir):
        os.makdir(output_dir)
    chipID = chip_id(chip_path)
    output_f = os.path.join(output_dir, "%s_jitter_0.jpg" % chipID)
    if os.path.isfile(output_f) and not force_rewrite:
        # print("%s already exists !" % output_f)
        return
    else:
        image = np.array(Image.open(chip_path))
        coeffs = face3d.subject_perspective_sighting_coefficients(coeff_path)
        ims = jitterer.multiple_random_jitters([image], coeffs, N_jitters)
        for i, im in enumerate(ims):
            output_f = os.path.join(output_dir, "%s_jitter_%s.jpg" % (chipID, i))
            Image.fromarray(im).save(output_f)
            # print("%s written! " % output_f)


def process_chunck(chunck):
    profile_jitterer = face3d.pose_jitterer_profile(pvr_data_path, 199, 29, "", min_yaw, max_yaw, egl_device)
    t_start = time.time()
    pid = os.getpid()
    for i, line in enumerate(chunck):
        process_line(line, profile_jitterer)
        if i % print_rate ==0  and i != 0:
            t_end = time.time()
            duration = t_end - t_start
            speed = float(print_rate) / duration
            print("[PID_%s]: [%s/%s] - jittering with speed %.3f im/s" % (pid, i, len(chunck), speed))
            t_start =time.time()

if __name__ == '__main__':
    lines = open(msceleb1m_coeffs_path, 'r').readlines()
    L = len(lines)
    chunksize = int(np.ceil(L / jobs))
    chunks = [lines[i: i + chunksize] for i in xrange(0, L, chunksize)]
    workers = Pool(processes=8)
    workers.map(process_chunck, chunks)
    workers.close()
    # process_chunck(chunks[0])
