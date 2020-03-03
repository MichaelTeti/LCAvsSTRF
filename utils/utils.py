from time import localtime
import numpy as np
from glob import glob
import os
import csv
import pvtools as pv
import torch


def read_csv(fname):
    items = []

    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader: items.append(row)

    return items


def check_if_dir_exists(directory, check_empty=False):
    if os.path.isdir(directory):
        exists = True
        if check_empty and len(os.listdir(directory)) == 0:
            exists = False
    else:
        exists = False

    return exists


def np_array_to_pt_tensor(array, use_cuda=True):
    tensor = torch.from_numpy(array).float()

    return tensor.cuda() if use_cuda else tensor


def get_sparsity(tensor):
    if type(tensor) == torch.Tensor:
        return torch.mean((tensor == 0).float()).item()
    elif type(tensor) == np.ndarray:
        return np.mean(np.float32(tensor == 0))


def get_sorted_files(dir, keyword=None, add_parent=False):
    if keyword:
        fnames = glob(os.path.join(dir, keyword))
        if not add_parent: fnames = [os.path.split(f)[1] for f in fnames]
    else:
        fnames = [os.path.join(dir, f) for f in os.listdir(dir)] if add_parent else os.listdir(dir)

    fnames.sort()

    return fnames



def walk_and_search(root_dir, keyword):
    fnames_all = []

    for root, _, files in os.walk(root_dir):
        fnames = get_sorted_files(root, keyword=keyword, add_parent=True)
        fnames_all += fnames

    return fnames_all



def write_csv(fname, items, mode='w'):
    with open(fname, mode) as f:
        writer = csv.writer(f, delimiter=',')

        for item in items:
            if type(item) != list:
                item = [item]

            writer.writerow(item)



def get_current_time():
    year, month, day, hour, min, sec, _, _, _ = localtime()
    min = min if len(str(min)) == 2 else '0' + str(min)
    sec = sec if len(str(sec)) == 2 else '0' + str(sec)
    return year, month, day, hour, min, sec


def bytescale(tensor):
    if type(tensor) == np.ndarray:
        tensor = tensor - np.amin(tensor)
        tensor = tensor / (np.amax(tensor) + 1e-6)
    elif type(tensor) == torch.Tensor:
        tensor = tensor - torch.min(tensor)
        tensor = tensor / (torch.max(tensor) + 1e-6)

    return tensor * 255


def get_fraction_active(filename):
    data = pv.readpvpfile(filename)
    data = np.array(data['values'])
    n, h, w, f = data.shape

    active = data != 0.0
    active_total = list(np.sum(active, (0,1,2)) / (n*h*w))

    feat_indices = list(range(len(active_total)))
    active_indices_sorted = [(x, y) for x, y in sorted(zip(active_total, feat_indices), reverse=True)]
    active_sorted = [x[0] for x in active_indices_sorted]
    feat_indices_sorted = [x[1] for x in active_indices_sorted]

    return active_sorted, feat_indices_sorted
