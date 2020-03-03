from argparse import ArgumentParser
import os, csv
from glob import glob
import pvtools as pv
import torch
import numpy as np
from progressbar import ProgressBar
from scipy.stats import pearsonr
from torch.nn.functional import cosine_similarity
from pywt import threshold
from scipy.spatial.distance import cosine

from utils import bytescale, \
                  np_array_to_pt_tensor, \
                  get_sparsity


parser = ArgumentParser()
parser.add_argument('m1_act_dir',
    type=str,
    help='Path to m1 activity files.')
parser.add_argument('m2_act_dir',
    type=str,
    help='Path to m2 activity files.')
parser.add_argument('filter',
    type=str,
    help="File prefix to filter by.")
parser.add_argument('txt_file_name',
    type=str,
    help='What to save the text file as.')
parser.add_argument('--thresh_type',
    type=str,
    choices=['hard', 'soft'],
    default='hard',
    help='Type of threshold.')
parser.add_argument('--downsample_b',
    type=int,
    default=1)
parser.add_argument('--downsample_h',
    type=int,
    default=1)
parser.add_argument('--downsample_w',
    type=int,
    default=1)
args = parser.parse_args()

m1_fnames = [f for f in os.listdir(args.m1_act_dir) if args.filter in f]
m2_fnames = [f for f in os.listdir(args.m2_act_dir) if args.filter in f]

m1_fnames.sort()
m2_fnames.sort()

m1_fnames = [os.path.join(args.m1_act_dir, f) for f in m1_fnames]
m2_fnames = [os.path.join(args.m2_act_dir, f) for f in m2_fnames]

timesteps = []
thresholds = list(np.arange(0, 1.1, 0.1))

if len(m2_fnames) != len(m1_fnames):
    assert(len(m2_fnames) == 1), \
        'The number of files in m1_act_dir ({}) is different than that in m2_act_dir ({}), \
        and the number of files in m2_act_dir is not 1.'.format(len(m1_fnames), len(m2_fnames))

    m2_fnames = m2_fnames * len(m1_fnames)

    write_header = True if not os.path.isfile(args.txt_file_name) else False
    f = open(args.txt_file_name, 'a')
    writer = csv.writer(f, delimiter=',')

    if write_header:
        writer.writerow(["LCA_Timestep", "STRF_Threshold", "Cossim", "LCA_Sparsity", "STRF_Sparsity", "Filter"])

    pbar = ProgressBar()
    for m1_fname, m2_fname in pbar(zip(m1_fnames, m2_fnames)):
        ts = int(os.path.split(m1_fname)[1].split('_')[1].split('.')[0].split('Checkpoint')[1])

        m1_act = pv.readpvpfile(m1_fname)['values'][::args.downsample_b, ::args.downsample_h, ::args.downsample_w, :]
        m2_act = pv.readpvpfile(m2_fname)['values'][::args.downsample_b, ::args.downsample_h, ::args.downsample_w, :]

        m1_act = bytescale(m1_act) / 255
        m2_act = bytescale(m2_act) / 255

        m1_act = np_array_to_pt_tensor(m1_act).flatten()

        for thresh in thresholds:
            m2_thresh = threshold(m2_act, thresh, mode=args.thresh_type)
            m2_thresh = np_array_to_pt_tensor(m2_thresh).flatten()

            m2_sparsity = get_sparsity(m2_thresh)
            m1_sparsity = get_sparsity(m1_act)

            cossim = cosine_similarity(m1_act, m2_thresh, dim=0).item()

            writer.writerow([ts, thresh, cossim, m1_sparsity, m2_sparsity, args.filter])

    f.close()
