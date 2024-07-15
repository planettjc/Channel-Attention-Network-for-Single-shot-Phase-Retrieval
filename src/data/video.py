import os

from data import common

import cv2
import numpy as np
import imageio

import torch
import torch.utils.data as data

class Video(data.Dataset):
    def __init__(self, args, name='Video', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.train = False
        self.do_eval = False
        self.benchmark = benchmark

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))
        self.vidcap = cv2.VideoCapture(args.dir_demo)
        self.n_frames = 0
        self.total_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        success, inp = self.vidcap.read()
        if success:
            self.n_frames += 1
            inp, = common.set_channel(inp, n_channels=self.args.n_colors)
            inp_t, = common.np2Tensor(inp, inten_range=self.args.inten_range)

            return inp_t, -1, '{}_{:0>5}'.format(self.filename, self.n_frames)
        else:
            vidcap.release()
            return None

    def __len__(self):
        return self.total_frames

