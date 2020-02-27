from argparse import ArgumentParser
import os, csv
from glob import glob
import pvtools as pv
import torch
import numpy as np
from progressbar import ProgressBar
from torch.nn.functional import cosine_similarity

from utils import walk_and_search, get_sorted_files, bytescale_patch_np


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
args = parser.parse_args()

m1_acts = pv.readpvpfile(args.model1_act_file)['values']
m2_acts = pv.readpvpfile(args.model2_act_file)['values']

m1_acts = bytescale_patch_np(m1_acts) / 255.
m2_acts = bytescale_patch_np(m2_acts) / 255.

print('m1 min and max are {} and {}.'.format(np.amin(m1_acts), np.amax(m1_acts)))
print('m2 min and max are {} and {}.'.format(np.amin(m2_acts), np.amax(m2_acts)))

m1_acts = torch.from_numpy(m1_acts).float().flatten()
m2_acts = torch.from_numpy(m2_acts).float().flatten()

if torch.cuda.is_available():
    m1_acts = m1_acts.cuda()
    m2_acts = m2_acts.cuda()

thresholds = np.arange(-1, 1.1, 0.01)

with open(args.txt_file_name, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Threshold', 'Cosine_Sim', 'STRF_Sparsity', 'LCA_Sparsity'])


    pbar = ProgressBar()
    for thresh in pbar(thresholds):
        m2_thresh = (m2_acts > thresh).float()
        m2_sparsity = torch.mean((m2_thresh == 0).float()).cpu().item()
        m2_thresh = m2_acts * m2_thresh
        m1_sparsity = torch.mean((m1_acts == 0).float()).cpu().item()
        cos_sim = cosine_similarity(m1_acts, m2_thresh, dim=0).cpu().item()
        writer.writerow([thresh, cos_sim, m2_sparsity, m1_sparsity])
