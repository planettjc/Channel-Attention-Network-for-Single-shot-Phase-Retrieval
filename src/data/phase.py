import os
from data import dataproc

class Phase(dataproc.DataProc):
    def __init__(self, args, name='Phase', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Phase, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_ph, names_in = super(Phase, self)._scan()
        names_ph = names_ph[self.begin - 1:self.end]
        names_in = names_in[self.begin - 1:self.end]

        return names_ph, names_in

    def _set_filesystem(self, dir_data):
        super(Phase, self)._set_filesystem(dir_data)
        self.apath = dir_data
        self.dir_ph = os.path.join(self.apath, 'Gt')
        self.dir_in = os.path.join(self.apath, 'Input')