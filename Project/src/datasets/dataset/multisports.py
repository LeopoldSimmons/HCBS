from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .base_dataset import BaseDataset


class MultiSports(BaseDataset):
    num_classes = 15

    def __init__(self, opt, mode):
        data_root = os.environ.get('HCBS_DATA_ROOT', os.path.join(opt.root_dir, 'data'))
        self.ROOT_DATASET_PATH = os.path.join(data_root, 'multisports')
        pkl_filename = 'multisports_GT.pkl'
        super(MultiSports, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'rgb-images', v, '{:0>5}.jpg'.format(i))

    def flowfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'brox-images', v, '{:0>5}.jpg'.format(i))
