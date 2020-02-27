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

from utils import bytescale_patch_np


parser = ArgumentParser()
parser.add_argument('model1_act_file',
    type=str,
    help="Path to the file with model 1's activity values.")
parser.add_argument('model2_act_file',
    type=str,
    help="Path to the file with model 2's activity values.")
parser.add_argument('txt_file_name',
    type=str,
    help='What to save the text file as.')
parser.add_argument('--thresh',
    type=float,
    default=0.25,
    help='Threshold for model2.')
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

m1_acts = pv.readpvpfile(args.model1_act_file)['values']
m2_acts = pv.readpvpfile(args.model2_act_file)['values']

m1_acts = m1_acts[::args.downsample_b, ::args.downsample_h, ::args.downsample_w, :]
m2_acts = m2_acts[::args.downsample_b, ::args.downsample_h, ::args.downsample_w, :]

m1_acts = bytescale_patch_np(m1_acts) / 255.
m2_acts = bytescale_patch_np(m2_acts) / 255.

print('m1 min and max are {} and {}.'.format(np.amin(m1_acts), np.amax(m1_acts)))
print('m2 min and max are {} and {}.'.format(np.amin(m2_acts), np.amax(m2_acts)))

m2_acts = m2_acts * np.float32(m2_acts > args.thresh)

m1_acts = m1_acts.flatten()
m2_acts = m2_acts.flatten()

r, p = pearsonr(m1_acts, m2_acts)
print('r = {}; p = {}.'.format(np.round(r, 2), np.round(p, 2)))

with open(args.txt_file_name, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["LCA", "STRF"])

    for m1_act, m2_act in zip(m1_acts, m2_acts):
        writer.writerow([m1_act, m2_act])
