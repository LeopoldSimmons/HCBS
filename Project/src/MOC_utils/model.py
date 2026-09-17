from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torchvision.models as models
import torch
import torch.nn as nn
import os
import tempfile
from MOC_utils.checkpoint import read_checkpoint, capture_rng, experiment_metadata

import torch.utils.model_zoo as model_zoo
from network.moc_net import MOC_Net
from network.moc_det import MOC_Det, MOC_Backbone


def create_model(arch, branch_info, head_conv, K, flip_test=False):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model = MOC_Net(arch, num_layers, branch_info, head_conv, K, flip_test=flip_test)
    return model


def create_inference_model(arch, branch_info, head_conv, K, flip_test=False):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    backbone = MOC_Backbone(arch, num_layers)
    branch = MOC_Det(backbone, branch_info, arch, head_conv, K, flip_test=flip_test)
    return backbone, branch


def load_model(model, model_path, optimizer=None, lr=None, ucf_pretrain=False):
    checkpoint = read_checkpoint(model_path, training=optimizer is not None)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    if ucf_pretrain:
        if optimizer is not None:
            raise ValueError('Transfer learning must create a fresh optimizer')
        state = {key: value for key, value in state.items()
                 if not key.startswith(('branch.hm.', 'branch.mov.'))}
        result = model.load_state_dict(state, strict=False)
        unexpected_missing = [key for key in result.missing_keys
                              if not key.startswith(('branch.hm.', 'branch.mov.'))]
        if unexpected_missing or result.unexpected_keys:
            raise ValueError('Unexpected transfer checkpoint keys: {} / {}'.format(unexpected_missing, result.unexpected_keys))
    else:
        model.load_state_dict(state, strict=True)
    if optimizer is None:
        return model
    if 'optimizer' not in checkpoint:
        raise ValueError('Resume requires optimizer state; use finetuning for weights-only checkpoints')
    optimizer.load_state_dict(checkpoint['optimizer'])
    if lr is not None:
        for group in optimizer.param_groups:
            group['lr'] = lr
    return model, optimizer, checkpoint['epoch'], checkpoint.get('best', float('-inf'))


def load_inference_model(backbone, branch, model_path):
    checkpoint = read_checkpoint(model_path)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    expected = set(backbone.state_dict()) | set(branch.state_dict())
    if set(state) != expected:
        raise ValueError('Inference checkpoint keys differ: missing={}, unexpected={}'.format(expected - set(state), set(state) - expected))
    backbone.load_state_dict({key: state[key] for key in backbone.state_dict()}, strict=True)
    branch.load_state_dict({key: state[key] for key in branch.state_dict()}, strict=True)
    return backbone, branch


def save_model(path, model, optimizer=None, epoch=0, best=float('-inf'), opt=None, training_state=None):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    data = {'epoch': epoch, 'best': best, 'state_dict': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
        data['rng'] = capture_rng()
        data['training_state'] = training_state or {}
    if opt is not None:
        data['experiment'] = experiment_metadata(opt)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=directory)
    os.close(fd)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_imagenet_pretrained_model(opt, model):
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
    }
    arch = opt.arch
    if arch == 'dla_34':
        print('load imagenet pretrained dla_34')
        model_url = model_urls['dla34']
        model_weights = model_zoo.load_url(model_url)

    elif arch.startswith('resnet'):
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        assert num_layers in (18, 34, 50, 101, 152)
        arch = arch[:arch.find('_')] if '_' in arch else arch

        print('load imagenet pretrained ', arch)
        url = model_urls['resnet{}'.format(num_layers)]
        model_weights = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))

    else:
        raise NotImplementedError
    new_state_dict = {}
    for key, value in model_weights.items():
        new_key = 'backbone.base.' + key
        new_state_dict[new_key] = value
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')
    model.load_state_dict(new_state_dict, strict=False)
    if opt.ninput > 1:
        convert2flow(opt.ninput, model)

    return model


def load_coco_pretrained_model(opt, model):
    if opt.arch == 'dla_34':
        print('load coco pretrained dla_34')
        model_path = '../experiment/modelzoo/coco_dla.pth'
    elif opt.arch == 'resnet_18':
        print('load coco pretrained resnet_18')
        model_path = '../experiment/modelzoo/coco_resdcn18.pth'
    elif opt.arch == 'resnet_101':
        print('load coco pretrained resnet_101')
        model_path = '../experiment/modelzoo/coco_resdcn101.pth'
    else:
        raise NotImplementedError
    checkpoint = read_checkpoint(model_path)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('wh'):
            new_key = 'branch.' + key
            new_state_dict[new_key] = value
        else:
            new_key = 'backbone.' + key
            new_state_dict[new_key] = value

    if 'resnet' in opt.arch:
        new_state_dict = convert_resnet_dcn(new_state_dict)

    print('load coco pretrained successfully')
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')

    model.load_state_dict(new_state_dict, strict=False)
    if opt.ninput > 1:
        convert2flow(opt.ninput, model)

    return model


def check_state_dict(load_dict, new_dict):
    # check loaded parameters and created model parameters
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            print('No param {}.'.format(k))
            new_dict[k] = load_dict[k]


def convert2flow(ninput, model):
    modules = list(model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    # first 7x7 conv : Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    conv_layer = modules[first_conv_idx]

    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    # kernel_size: [16, 3, 7, 7]
    kernel_size = params[0].size()
    # new_kernel_size: [16, 3*ninput, 7, 7]
    new_kernel_size = kernel_size[:1] + (3 * ninput, ) + kernel_size[2:]

    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(3 * ninput, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convlution layer

    setattr(container, layer_name, new_conv)
    print('load pretrained model to flow input')
    return model


def convert_resnet_dcn(state_dict):
    new_state_dict = {}
    for k in state_dict:
        if k.startswith('backbone.deconv_layer'):
            new_k = 'backbone.deconv_layer.deconv_layers' + k.split('deconv_layers')[1]
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    return new_state_dict
