import os

from data import common
from data import dataproc

import numpy as np

import torch
import torch.utils.data as data
import glob
import pdb

class Benchmark(dataproc.DataProc):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True)

    def _scan(self):
        list_ph = []
        list_in = []
        for entry in os.scandir(self.dir_ph):
            filename = os.path.splitext(entry.name)[0]
            if "PH" in filename:
                list_ph.append(os.path.join(self.dir_ph, filename + self.ext))
        #pdb.set_trace()
        for entry in os.scandir(self.dir_in):
            filename = os.path.splitext(entry.name)[0]
            if "IN" in filename:
                list_in.append(os.path.join(self.dir_in, filename + self.ext))

        list_ph.sort()
        list_in.sort()

        return list_ph, list_in

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.all_files = glob.glob(os.path.join(self.apath, 'Gt', "*.png"))
        #self.dir_in = os.path.join(dir_data, self.name, 'Test/3')
        #self.dir_ph = os.path.join(dir_data, self.name, 'Test/3')
        self.dir_in = os.path.join(dir_data, self.name, 'Input')
        self.dir_ph = os.path.join(dir_data, self.name, 'Gt')
        #self.dir_in = os.path.join(self.apath, 'IN_bicubic')
        self.ext = '.png'