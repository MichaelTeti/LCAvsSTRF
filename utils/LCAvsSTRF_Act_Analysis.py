import os
from argparse import ArgumentParser

from utils import write_csv, get_sorted_files
from SingleCkptAnalysis import SingleCkptAnalysis






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('LCA_ckpt_dir',
        type=str,
        help='Path to the LCA ckpts.')
    parser.add_argument('STRF_ckpt_dir',
        type=str,
        help='Path to the STRF ckpts.')
    parser.add_argument('--csv_fname',
        type=str,
        default='act.txt',
        help='Name to save the csv files as. Default is act.txt.')
    args = parser.parse_args()

    assert(os.path.isdir(args.LCA_ckpt_dir) and os.listdir(args.LCA_ckpt_dir) != []), \
        'LCA_ckpt_dir {} does not exist.'.format(args.LCA_ckpt_dir)
    assert(os.path.isdir(args.STRF_ckpt_dir) and os.listdir(args.STRF_ckpt_dir) != []), \
        'STRF_ckpt_dir {} does not exist.'.format(args.STRF_ckpt_dir)

    LCA_ckpts = get_sorted_files(args.LCA_ckpt_dir, add_parent=True)
    STRF_ckpts = get_sorted_files(args.STRF_ckpt_dir, add_parent=True)

    
