from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle

from opts import opts
from datasets.init_dataset import switch_dataset
from detector.normal_moc_det import MOCDetector
import random
# MODIFY FOR PYTORCH 1+
# cv2.setNumThreads(0)
GLOBAL_SEED = 317

from datasets.io_utils import read_image, load_text
from MOC_utils.cache import prepare_cache, atomic_pickle

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.vlist = dataset.video_list
        self.gttubes = dataset._gttubes
        self.nframes = dataset._nframes
        self.textfile = dataset.textfile
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        for v in self.vlist:
            for i in range(1, 1 + self.nframes[v] - self.opt.K + 1):
                if opt.redo or not os.path.exists(self.outfile(v, i)):
                    self.indices += [(v, i)]

    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        pkl_file = []
        txt_tensor = []

        if self.opt.rgb_model != '':
            images = [read_image(self.imagefile(v, frame + i)) for i in range(self.opt.K)]
            images = self.pre_process_func(images)

            if self.opt.modality == 'multimodal':
                txt_tensor = [load_text(self.textfile(v, frame + i), self.opt.text_format,
                                        self.input_h, self.input_w) for i in range(self.opt.K)]
                if self.opt.flip_test:
                    # Text maps are semantic representations, not spatial image pixels.
                    txt_tensor = txt_tensor + [tensor.copy() for tensor in txt_tensor]

        if self.opt.flow_model != '':
            flows = [read_image(self.flowfile(v, min(frame + i, self.nframes[v]))) for i in range(self.opt.K + self.opt.ninput - 1)]
            flows = self.pre_process_func(flows, is_flow=True, ninput=self.opt.ninput)

        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

        return {'outfile': outfile, 'textdata': txt_tensor, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}}
        # return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def normal_inference(opt, drop_last=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.backends.cudnn.benchmark = not opt.deterministic

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, opt.eval_split)
    prepare_cache(opt, dataset, 'normal')
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process)
    total_num = len(prefetch_dataset)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    print('inference chunk_sizes:', opt.chunk_sizes)
    for iter, data in enumerate(data_loader):
        outfile = data['outfile']

        detections = detector.run(data)

        for i in range(len(outfile)):
            atomic_pickle(outfile[i], detections[i])


        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    return total_num
