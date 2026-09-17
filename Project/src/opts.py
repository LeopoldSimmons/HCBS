from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--modality', choices=['multimodal', 'visual_only'], default='multimodal')
        self.parser.add_argument('--text_format', choices=['feature_map', 'legacy_image'], default='feature_map')
        self.parser.add_argument('--text_root', default='', help='Root containing video/frame.npy files')
        self.parser.add_argument('--text_revision', default='', help='Immutable feature revision for legacy exports without a manifest')
        self.parser.add_argument('--data_root', default=os.environ.get('HCBS_DATA_ROOT', ''))
        self.parser.add_argument('--split_manifest', default='')
        self.parser.add_argument('--eval_split', choices=['val', 'test'], default='test')
        self.parser.add_argument('--num_classes', type=int, default=None)
        self.parser.add_argument('--save_dir', default='')
        self.parser.add_argument('--resume', action='store_true', help='Restore full trusted training checkpoint')
        self.parser.add_argument('--allow_legacy_checkpoint', action='store_true', help='Allow missing experiment metadata, not missing weights')
        self.parser.add_argument('--deterministic', action='store_true')
        self.parser.add_argument('--max_objs', type=int, default=128)
        self.parser.add_argument('--dcn_backend', choices=['torchvision', 'legacy'], default='torchvision')

        # basical experiment settings
        self.parser.add_argument('--task', default='MOC_train',
                                 help='current task')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--model_name', default='None',
                                 help='corrent model name')
        self.parser.add_argument('--load_model', default='',
                                 help='path to load model')
        self.parser.add_argument('--rgb_model', default='',
                                 help='path to rgb model')
        self.parser.add_argument('--flow_model', default='',
                                 help='path to flow model')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')
        self.parser.add_argument('--ninput', type=int, default=1,
                                 help='length of input, 1 for rgb, 5 for flow by default')

        # model seeting
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resnet_18 | resnet_101 | dla_34')
        self.parser.add_argument('--set_head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      'default setting is 256 for dla and 256 for resnet(except for wh branch) ')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        self.parser.add_argument('--K', type=int, default=7,
                                 help='length of action tube')

        # system settings
        self.parser.add_argument('--gpus', default='0,1',
                                 help='visible gpu list, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=16,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--batch_size', type=int, default=128,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu. -1 by default')

        # learning rate settings
        self.parser.add_argument('--lr', type=float, default=5e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='6,8',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')

        # dataset seetings
        self.parser.add_argument('--dataset', default='ucf101',
                                 help='ucf101 | hmdb')
        self.parser.add_argument('--split', type=int, default=1,
                                 help='1 split for UCF101, 3 splits for JHMDB')
        self.parser.add_argument('--resize_height', type=int, default=288,
                                 help='input image height')
        self.parser.add_argument('--resize_width', type=int, default=288,
                                 help='input image width')

        # training settings
        self.parser.add_argument('--pretrain_model', default='coco',
                                 help='training pretrain_model, coco | imagenet')
        self.parser.add_argument('--ucf_pretrain', action='store_true',
                                 help='use ucf pretrain for jhmdb')

        self.parser.add_argument('--auto_stop', action='store_true',
                                 help='select best checkpoint on explicit validation AP (no test selection or optimizer rollback)')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save each epoch training model')
        self.parser.add_argument('--val_epoch', action='store_true',
                                 help='val after each epoch')
        self.parser.add_argument('--visual_per_inter', type=int, default=100,
                                 help='iter for draw loss by tensorboardX')

        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 help='strat epoch, used for recover experiment')
        self.parser.add_argument('--pin_memory', action='store_true',
                                 help='set pin_memory True')

        # loss ratio settings
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for center heatmaps.')
        self.parser.add_argument('--mov_weight', type=float, default=1,
                                 help='loss weight for moving offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bbox regression.')

        # inference settings
        self.parser.add_argument('--allow_legacy_predictions', action='store_true')
        self.parser.add_argument('--redo', action='store_true',
                                 help='redo for count APs')
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--N', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--inference_dir', default='tmp',
                                 help='directory for inferencing')
        self.parser.add_argument('--th', type=float, default=0.5,
                                 help='threshod for ACT.py')

        # fusion settings for rgb and flow
        self.parser.add_argument('--hm_fusion_rgb', type=float, default=0.5,
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--mov_fusion_rgb', type=float, default=0.8,
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--wh_fusion_rgb', type=float, default=0.8,
                                 help='rgb : th, flow: 1 - th')

        # log
        self.parser.add_argument('--print_log', action='store_true',
                                 help='print log info')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.set_head_conv != -1:
            opt.head_conv = opt.set_head_conv
        elif 'dla' in opt.arch:
            opt.head_conv = 256
        elif 'resnet' in opt.arch:
            opt.head_conv = 256

        if opt.ninput == 1 and opt.flow_model != '':
            opt.ninput = 5

        opt.mean = [0.40789654, 0.44719302, 0.47026115]
        opt.std = [0.28863828, 0.27408164, 0.27809835]
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)

        if opt.modality == 'multimodal' and ('dla' not in opt.arch or opt.ninput > 1):
            self.parser.error('Multimodal training/inference requires DLA RGB; use --modality visual_only for flow or ResNet')
        if opt.resume and (not opt.load_model or opt.ucf_pretrain):
            self.parser.error('--resume requires --load_model and cannot be combined with --ucf_pretrain')
        if (opt.val_epoch or opt.auto_stop) and not opt.split_manifest:
            self.parser.error('Validation/model selection requires --split_manifest with disjoint train/val/test')
        if opt.K < 1 or opt.batch_size < 1 or opt.max_objs < 1 or opt.visual_per_inter < 1:
            self.parser.error('K, batch_size, max_objs and visual_per_inter must be positive')
        if opt.down_ratio != 4 or min(opt.resize_height, opt.resize_width) < 32 or opt.resize_height % 32 or opt.resize_width % 32:
            self.parser.error('Use down_ratio=4 and image dimensions divisible by 32')
        if opt.task in ('stream', 'speed_test') and opt.modality != 'visual_only':
            self.parser.error('Streaming and speed_test are visual_only; use --task normal for multimodal inference')
        if opt.batch_size < len(opt.gpus) or any(size < 1 for size in opt.chunk_sizes):
            self.parser.error('Batch size/chunk sizes must allocate at least one example per GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        os.environ['HCBS_DCN_BACKEND'] = opt.dcn_backend
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.data_root = opt.data_root or os.path.join(opt.root_dir, 'data')
        opt.save_dir = opt.save_dir or os.path.join(opt.root_dir, 'experiments', opt.exp_id)
        opt.log_dir = os.path.join(opt.save_dir, 'logs_tensorboardX')

        return opt

    def update_dataset(self, opt, dataset):
        opt.num_classes = dataset.num_classes if opt.num_classes is None else opt.num_classes
        opt.branch_info = {'hm': opt.num_classes,
                           'mov': 2 * opt.K,
                           'wh': 2 * opt.K}
        return opt
