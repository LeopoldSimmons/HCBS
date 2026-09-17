"""Checkpoint metadata and explicit trusted resume support."""
import random
import warnings
from pathlib import Path

import numpy as np
import torch


def read_checkpoint(path, training=False):
    # Full resume includes Python/NumPy RNG state: only resume trusted local files.
    return torch.load(path, map_location='cpu', weights_only=not training)


def experiment_metadata(opt):
    from MOC_utils.cache import sha256_file
    return {key: getattr(opt, key) for key in (
        'dataset', 'split', 'num_classes', 'arch', 'head_conv', 'K', 'ninput', 'modality',
        'text_format', 'resize_height', 'resize_width', 'dcn_backend')} | {
        'split_manifest_sha256': sha256_file(opt.split_manifest) if opt.split_manifest else None,
        'text_revision': text_revision(opt)}


def text_revision(opt):
    if opt.modality == 'visual_only':
        return None
    from MOC_utils.cache import sha256_file
    dataset_folder = {'ucf101': 'UCF101_v2', 'hmdb': 'JHMDB', 'multisports': 'multisports'}[opt.dataset]
    root = Path(opt.text_root) if opt.text_root else Path(opt.data_root) / dataset_folder / 'numpys'
    manifest = root / 'features.json'
    if manifest.is_file():
        return sha256_file(manifest)
    if opt.text_revision:
        return opt.text_revision
    raise ValueError('Text features need features.json or an explicit immutable --text_revision')


def check_metadata(checkpoint, opt, input_kind=None):
    saved = checkpoint.get('experiment')
    if saved is None:
        if not opt.allow_legacy_checkpoint:
            raise ValueError('Checkpoint lacks experiment metadata; use --allow_legacy_checkpoint only after confirming its protocol')
        warnings.warn('Loading legacy checkpoint without experiment metadata')
        return
    if getattr(opt, 'visualization', False):
        expected = {key: getattr(opt, key) for key in ('num_classes', 'arch', 'head_conv', 'K', 'ninput', 'modality', 'resize_height', 'resize_width', 'dcn_backend')}
    else:
        expected = experiment_metadata(opt)
    if input_kind == 'rgb':
        expected['ninput'] = 1
    differences = {k: (saved.get(k), value) for k, value in expected.items() if saved.get(k) != value}
    if differences:
        raise ValueError('Checkpoint experiment mismatch: {}'.format(differences))


def capture_rng():
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, values, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
