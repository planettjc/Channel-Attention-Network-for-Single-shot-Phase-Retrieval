from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))
            # print(datasets[0].__len__())
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Val','Test']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
