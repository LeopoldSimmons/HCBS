from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
import time

import numpy as np
import torch

from opts import opts
from inference.normal_inference import normal_inference
from inference.stream_inference import stream_inference


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    shutil.rmtree('tmp', ignore_errors=True)
    opt = opts().parse()
    t1 = time.time()
    set_seed(opt.seed)
    if opt.task == 'normal':
        normal_inference(opt)
    elif opt.task == 'stream':
        stream_inference(opt)
    elif opt.task == 'speed_test':
        from inference.speed_test import speed_test_stream_inference
        speed_test_stream_inference(opt)
    else:
        raise NotImplementedError
    print('total_time: ', (time.time() - t1) / 60)
