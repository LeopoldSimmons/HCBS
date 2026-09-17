"""Explicit, disjoint video partitions and class contracts."""
import json
from pathlib import PurePosixPath


def configure_protocol(dataset, opt):
    if opt.split_manifest:
        with open(opt.split_manifest, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('dataset') != opt.dataset or manifest.get('split') != opt.split:
            raise ValueError('Split manifest dataset/split does not match options')
        partitions = {name: manifest[name] for name in ('train', 'val', 'test')}
    else:
        partitions = {'train': dataset._train_videos[opt.split - 1],
                      'val': [], 'test': dataset._test_videos[opt.split - 1]}
    seen = set()
    for name, videos in partitions.items():
        if not isinstance(videos, list) or any(not isinstance(v, str) for v in videos):
            raise ValueError('{} partition must be a list of video IDs'.format(name))
        if len(videos) != len(set(videos)) or seen.intersection(videos):
            raise ValueError('Duplicate or overlapping video IDs in {}'.format(name))
        for video in videos:
            path = PurePosixPath(video)
            if path.is_absolute() or '..' in path.parts or '\\' in video:
                raise ValueError('Unsafe video ID: {}'.format(video))
            if video not in dataset._nframes or video not in dataset._gttubes:
                raise ValueError('Unknown video ID in {}: {}'.format(name, video))
        seen.update(videos)
    if len(dataset.labels) != opt.num_classes:
        raise ValueError('GT has {} labels but model has {} classes; use --num_classes and a matching GT file'.format(
            len(dataset.labels), opt.num_classes))
    for video in seen:
        if any(label < 0 or label >= opt.num_classes for label in dataset._gttubes[video]):
            raise ValueError('Unmapped class ID in {}'.format(video))
    dataset.partitions = partitions
    selected = dataset.mode
    if selected not in partitions:
        raise ValueError('Unknown dataset mode: {}'.format(selected))
    if not partitions[selected]:
        raise ValueError('Empty {} partition; validation requires an explicit --split_manifest'.format(selected))
    dataset.video_list = partitions[selected]
