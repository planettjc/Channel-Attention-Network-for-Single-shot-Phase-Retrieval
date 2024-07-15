import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import pdb
#import pdb

class DataProc(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_ph, list_in = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_ph, self.images_in = list_ph, list_in
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_ph.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_in.replace(self.apath, path_bin),
                exist_ok=True
            )
            
            self.images_ph, self.images_in = [], []
            for h in list_ph:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_ph.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for l in list_in:
                b = l.replace(self.apath, path_bin)
                b = b.replace(self.ext[1], '.pt')
                self.images_in.append(b)
                self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_ph)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_ph = sorted(
            glob.glob(os.path.join(self.dir_ph, '*' + self.ext[0]))
        )
        names_in = []
        for f in names_ph:
            filename,_ = os.path.splitext(os.path.basename(f))[0].split('_')
            names_in.append(os.path.join(self.dir_in, '{}{}{}'.format(filename, '_IN', self.ext[1])))

        return names_ph, names_in

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_ph = os.path.join(self.apath, 'Gt')
        self.dir_in = os.path.join(self.apath, 'Input')
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        inp, ph, filename = self._load_file(idx)
        pair = self.get_patch(inp, ph)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, inten_range=self.args.inten_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_ph) * self.repeat
        else:
            return len(self.images_ph)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_ph)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_ph = self.images_ph[idx]
        f_in = self.images_in[idx]
        #print('！！!！!！!!！',f_in)
        #pdb.set_trace()

        filename, _ = os.path.splitext(os.path.basename(f_ph))
        if self.args.ext == 'img' or self.benchmark:
            ph = imageio.imread(f_ph)
            inp = imageio.imread(f_in)
        elif self.args.ext.find('sep') >= 0:
            with open(f_ph, 'rb') as _f:
                ph = pickle.load(_f)
            with open(f_in, 'rb') as _f:
                inp = pickle.load(_f)

        return inp, ph, filename

    def get_patch(self, inp, ph):
        if self.train:
            inp, ph = common.get_patch(
                inp, ph,
                patch_size=self.args.patch_size
            )
            #print(ph.shape)
            if not self.args.no_augment: inp, ph = common.augment(inp, ph)
        else:
            ih, iw = inp.shape[:2]
            ph = ph[0:ih, 0:iw]

        return inp, ph