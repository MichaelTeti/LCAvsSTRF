from argparse import ArgumentParser
import os, csv
from glob import glob
import pvtools as pv
import torch
import numpy as np
from progressbar import ProgressBar
from torch.nn.functional import cosine_similarity
import pandas as pd
from scipy.stats import pearsonr
from pywt import threshold

from utils import bytescale, get_sparsity


parser = ArgumentParser()
parser.add_argument('act_file',
    type=str,
    help="Path to the file with activity values.")
parser.add_argument('txt_file_name',
    type=str,
    help='What to save the text file as.')
parser.add_argument('--key',
    type=str,
    default=None,
    help='Key for this output in the text file.')
parser.add_argument('--thresh',
    type=float,
    default=0.25,
    help='Threshold for activations.')
parser.add_argument('--thresh_type',
    type=str,
    choices=['soft', 'hard'],
    default='hard',
    help='Threshold type.')
parser.add_argument('--downsample_b',
    type=int,
    default=1,
    help='How much to downsample the batch.')
parser.add_argument('--downsample_h',
    type=int,
    default=1,
    help='How much to downsample the height.')
parser.add_argument('--downsample_w',
    type=int,
    default=1,
    help='How much to downsample the width.')
args = parser.parse_args()

acts = pv.readpvpfile(args.act_file)['values']
acts = acts[::args.downsample_b, ::args.downsample_h, ::args.downsample_w, :]

acts = bytescale(acts) / 255.

print('min and max are {} and {}.'.format(np.amin(acts), np.amax(acts)))

if args.thresh != 0:
    acts = threshold(acts, args.thresh, mode=args.thresh_type)

acts = acts.flatten()

print('sparsity is {}.'.format(get_sparsity(acts)))

write_header = False if os.path.isfile(args.txt_file_name) else True

with open(args.txt_file_name, 'a') as f:
    writer = csv.writer(f, delimiter=',')
    if write_header: writer.writerow(["Act", "Key"])

    for act in acts:
        writer.writerow([act, args.key])
