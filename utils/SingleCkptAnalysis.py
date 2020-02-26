import pvtools as pv
import numpy as np
from visdom import Visdom
from imageio import imwrite, mimsave
import os
import matplotlib.pyplot as plt
from time import sleep, localtime
from glob import glob
from shutil import rmtree
import csv
from argparse import ArgumentParser
from utils import get_current_time, get_sorted_files, bytescale_patch_np


class SingleCkptAnalysis():
    def __init__(self, ckpt_dir, analysis_dir, check_frequency=120,
                 delete_old_analyses=True, ckpt_freq=4000, weight_gif=True,
                 recon_gif=True, model_layer_name='S1', probe_file_dir=None):
        self.vis = Visdom()
        self.check_frequency = check_frequency
        self.ckpt_dir = ckpt_dir
        self.analysis_dir = analysis_dir
        self.latest_analysis = self.get_latest_analyses(self.analysis_dir)
        self.delete_old_analyses = delete_old_analyses
        self.ckpt_freq = ckpt_freq
        self.weight_gif = weight_gif
        self.recon_gif = recon_gif
        self.model_layer_name = model_layer_name
        self.probe_file_dir = probe_file_dir

        print('[INFO] CHECKPOINT DIR: {}'.format(self.ckpt_dir))
        print('[INFO] ANALYSIS DIR: {}'.format(self.analysis_dir))
        print('[INFO] LATEST EXISTING ANALYSIS: {}'.format(self.latest_analysis))



    def get_latest_analyses(self, dir):
        if not os.path.isdir(dir): os.mkdir(dir)
        existing = get_sorted_files(dir, 'analysis*', add_parent=False)

        return existing[-1] if any(existing) else None


    def get_current_ckpt(self):
        current_ckpts = get_sorted_files(self.ckpt_dir, add_parent=False)
        latest_ckpt = current_ckpts[-1]
        latest_ckpt_path = os.path.join(self.ckpt_dir, latest_ckpt)
        latest_ckpt_num = latest_ckpt.split('Checkpoint')[-1]

        return latest_ckpt, latest_ckpt_path, latest_ckpt_num


    def montage_weights(self, ckpt_dir, save_dir, sorted_indices):
        weight_filenames = get_sorted_files(ckpt_dir, keyword='*_W.pvp', add_parent=True)
        #weight_filenames = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if '_W.pvp' in f]
        #weight_filenames.sort()
        save_dir = os.path.join(save_dir, 'Weights')
        os.mkdir(save_dir)
        gif = [] if self.weight_gif else None

        for i_filename, weight_filename in enumerate(weight_filenames):
            data = pv.readpvpfile(weight_filename)
            weights = data['values']
            weights = weights[0, 0, :, :, :, 0]
            f, h, w = weights.shape
            weights = weights[sorted_indices, ...]
            gridh, gridw = h * int(np.ceil(np.sqrt(f))), w * int(np.ceil(np.sqrt(f)))
            grid = np.zeros([gridh, gridw])
            count = 0

            for i_h in range(0, gridh, h):
                for i_w in range(0, gridw, w):
                    if count < f:
                        grid[i_h:i_h+h, i_w:i_w+w] = bytescale_patch_np(weights[count, ...])
                        count += 1

            grid[::h, :] = 255.
            grid[:, ::w] = 255.

            if not self.weight_gif:
                fig_name = os.path.split(weight_filename)[1][:-4]
                imwrite(os.path.join(save_dir, fig_name + '.png'), np.uint8(grid))
            else:
                gif.append(np.uint8(grid))
                mimsave(os.path.join(save_dir, 'weights.gif'), gif, fps=5)



    def plot_recs(self, ckpt_dir, save_dir):
        rec_paths = get_sorted_files(ckpt_dir, keyword='Frame*Recon_A.pvp', add_parent=True)
        if rec_paths == []: return
        save_dir = os.path.join(save_dir, 'Recons')
        gifs = {} if self.recon_gif else None

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i_rec_path, rec_path in enumerate(rec_paths):
            i_input_frame = int(''.join(filter(str.isdigit, os.path.split(rec_path)[1])))
            input_filename = 'Frame{}_A.pvp'.format(i_input_frame)
            input_path = os.path.join(ckpt_dir, input_filename)
            input_batch, rec_batch = pv.readpvpfile(input_path)['values'], pv.readpvpfile(rec_path)['values']
            n = input_batch.shape[0]
            frame_save_dir = os.path.join(save_dir, 'Frame{}'.format(i_input_frame)) if self.recon_gif else save_dir

            if not os.path.isdir(frame_save_dir) and not self.recon_gif:
                os.mkdir(frame_save_dir)

            for i_example, (input_ex, rec_ex) in enumerate(zip(input_batch, rec_batch)):
                if i_example not in list(gifs.keys()):
                    gifs[i_example] = []

                input_ex, rec_ex = input_ex[..., 0], rec_ex[..., 0]
                input_scaled, rec_scaled = bytescale_patch_np(input_ex), bytescale_patch_np(rec_ex)

                if np.sum(rec_scaled) == 0 and int(''.join([c for c in self.latest_analysis if c.isdigit()])) != 0:
                    print('[WARNING] BATCH {} EXPLODED'.format(os.path.split(rec_path)[1]))

                divider = np.zeros([input_scaled.shape[0], int(input_scaled.shape[1]*0.05)])
                pane = np.uint8(np.concatenate((input_scaled, divider, rec_scaled), 1))

                if not self.recon_gif:
                    imwrite(os.path.join(frame_save_dir, 'Example{}Input.png'.format(i_example)), pane)
                else:
                    gifs[i_example].append(pane)

        [mimsave(os.path.join(save_dir, 'Recon_{}.gif'.format(k)), gifs[k], fps=15) for k in list(gifs.keys())]



    def get_fraction_active_total(self, ckpt_dir, save_dir):
        filename = os.path.join(ckpt_dir, self.model_layer_name + '_A.pvp')
        data = pv.readpvpfile(filename)
        data = np.array(data['values'])
        n, h, w, f = data.shape

        active = data != 0.0
        active_total = list(np.sum(active, (0,1,2)) / (n*h*w))

        feat_indices = list(range(len(active_total)))
        active_indices_sorted = [(x, y) for x, y in sorted(zip(active_total, feat_indices), reverse=True)]
        active_sorted = [x[0] for x in active_indices_sorted]
        feat_indices_sorted = [x[1] for x in active_indices_sorted]
        opts = dict(xlabel='Feature Number', ylabel='Fraction Active', title='Activations' + self.latest_analysis.split('-')[1])
        self.vis.bar(active_sorted, win='frac_act_total', opts=opts)

        return feat_indices_sorted, active_sorted



    def plot_energy(self, save_dir):
        if not self.probe_file_dir: return
        files = get_sorted_files(self.probe_file_dir, keyword='EnergyProbe_*', add_parent=True)
        if files == []: return
        save_dir = os.path.join(save_dir, 'Energy')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        for i_file, file in enumerate(files):
            name = os.path.split(file)[1].split('.')[0] + '.png'
            data = np.genfromtxt(file, delimiter=',', skip_header=1)
            end_x = [t for t in data[:, 0]  if t % self.ckpt_freq == 0]
            end_x = int(max(end_x)) if end_x != [] else 0
            start_x = end_x - self.ckpt_freq if end_x != 0 else 0
            end_x = data.shape[0] if end_x == 0 else end_x
            data = data[start_x:end_x, :]

            fig = plt.figure(figsize=(20, 15))
            subplot = fig.add_subplot(111)
            subplot.set_ylabel('Energy')
            subplot.set_xlabel('Timestep')
            subplot.plot(data[:, 0], data[:, -1])
            plt.savefig(os.path.join(save_dir, name))
            plt.close()



    def plot_adaptivetimescales(self, save_dir):
        if not self.probe_file_dir: return
        files = get_sorted_files(self.probe_file_dir, keyword='AdaptiveTimeScales*', add_parent=True)
        if files == []: return
        save_dir = os.path.join(save_dir, 'AdaptiveTimeScales')
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        for i_file, file in enumerate(files):
            name = os.path.split(file)[1].split('.')[0]

            with open(file, 'r+') as txtfile:
                reader = csv.reader(txtfile, delimiter=',')
                timescale_data = {'Timescale': [], 'TimescaleTrue': [], 'TimescaleMax': [], 'Time': []}

                for i_row, row in enumerate(reader):
                    if len(row) == 1:
                        timescale_data['Time'].append(float(row[0].split(' = ')[-1]))
                    else:
                        timescale_data['Timescale'].append(float(row[1].split(' = ')[-1]))
                        timescale_data['TimescaleTrue'].append(float(row[2].split(' = ')[-1]))
                        timescale_data['TimescaleMax'].append(float(row[3].split(' = ')[-1]))

            end_x = [t for t in timescale_data['Time']  if t % self.ckpt_freq == 0]
            end_x = int(max(end_x)) if end_x != [] else 0
            start_x = end_x - self.ckpt_freq if end_x != 0 else 0
            end_x = len(timescale_data['Time']) if end_x == 0 else end_x

            for key in list(timescale_data.keys())[:-1]:
                fig = plt.figure(figsize=(20, 15))
                subplot = fig.add_subplot(111)
                subplot.set_ylabel(key)
                subplot.set_xlabel('Time')
                subplot.plot(timescale_data['Time'][start_x:end_x], timescale_data[key][start_x:end_x])
                plt.savefig(os.path.join(save_dir, name + '_' + key + '.png'))
                plt.close()



    def analyze(self):
        while True:
            _, current_ckpt_dir, current_ckpt_num = self.get_current_ckpt()

            if 'analysis-' + current_ckpt_num != self.latest_analysis:
                year, month, day, hour, min, sec = get_current_time()
                print('[INFO] FOUND A NEW CHECKPOINT: {} ({}/{}/{} {}:{}:{} EST)' \
                      .format(current_ckpt_num, month, day, year, hour, min, sec))
                self.latest_analysis = 'analysis-' + current_ckpt_num
                save_dir = os.path.join(self.analysis_dir, self.latest_analysis)
                os.mkdir(save_dir)
                act_fname = os.path.join(current_ckpt_dir, self.model_layer_name + '_A.pvp')

                sorted_feat_indices, sorted_activations = self.get_fraction_active_total(current_ckpt_dir, save_dir)
                self.montage_weights(current_ckpt_dir, save_dir, sorted_feat_indices)
                self.plot_recs(current_ckpt_dir, save_dir)
                self.plot_energy(save_dir)
                self.plot_adaptivetimescales(save_dir)

                print('[INFO] ANALYSIS {} WRITE COMPLETE.'.format(current_ckpt_num))

                if self.delete_old_analyses and len(glob(os.path.join(self.analysis_dir, 'analysis-*'))) > 1:
                    print('[INFO] REMOVING OLD ANALYSIS FILES.')
                    [rmtree(f) for f in glob(os.path.join(self.analysis_dir, 'analysis-*')) if f != save_dir]

            sleep(self.check_frequency)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ckpt_dir',
                        type=str,
                        help='Path to the checkpoint files.')
    parser.add_argument('analysis_dir',
                        type=str,
                        help='Path to save analysis files.')
    parser.add_argument('--check_frequency',
                        type=int,
                        default=600,
                        help='How often to check for a new checkpoint file. Default is 600s.')
    parser.add_argument('--ckpt_frequency',
                        type=int,
                        default=1000,
                        help='How long the display period is. Default 1000 timesteps.')
    parser.add_argument('--model_layer_name',
                        type=str,
                        default='S1',
                        help='Model layer name.')
    parser.add_argument('--probe_file_dir',
                        type=str,
                        help='Path to directory containing the probe files.')

    args = parser.parse_args()

    analyzer = SingleCkptAnalysis(args.ckpt_dir,
                                  args.analysis_dir,
                                  args.check_frequency,
                                  weight_gif=True,
                                  recon_gif=True,
                                  ckpt_freq=args.ckpt_frequency,
                                  model_layer_name=args.model_layer_name,
                                  probe_file_dir=args.probe_file_dir)

    analyzer.analyze()
