from argparse import ArgumentParser
from imageio import imwrite
import pvtools as pv
import numpy as np
import os
from progressbar import ProgressBar
from pywt import threshold

from utils import bytescale, check_if_dir_exists



parser = ArgumentParser()
parser.add_argument('act_file_path',
    type=str,
    help="Directory containing one or multiple activation files.")
parser.add_argument('output_dir',
    type=str,
    help='Directory where the images will be saved.')
parser.add_argument('--thresh',
    type=float,
    default=0.0,
    help='Threshold if desired for the activity values.')
parser.add_argument('--thresh_type',
    type=str,
    choices=['hard', 'soft'],
    default='hard',
    help='Type of threshold.')
parser.add_argument('--n_batch',
    type=str,
    default='all')
parser.add_argument('--binary',
    default=False,
    action='store_true')
args = parser.parse_args()


name = os.path.splitext(os.path.split(args.act_file_path)[1])[0]
if not check_if_dir_exists(args.output_dir): os.mkdir(args.output_dir)

act_data = pv.readpvpfile(args.act_file_path)['values']
if args.n_batch != 'all': act_data = act_data[:int(args.n_batch), ...]
act_data = bytescale(act_data)
if args.thresh != 0: act_data = threshold(act_data, args.thresh, mode=args.thresh_type)
act_data = np.uint8(act_data)

pbar = ProgressBar()
for i_sample, sample in pbar(enumerate(act_data)):
    for i_feature in range(sample.shape[-1]):
        feat_map = sample[..., i_feature]
        if args.binary:
            feat_map[feat_map > np.mean(feat_map)] = 255
            feat_map[feat_map < np.mean(feat_map)] = 0

        save_path = os.path.join(args.output_dir, name + '_b{}f{}.png'.format(i_sample, i_feature))
        imwrite(save_path, feat_map)
