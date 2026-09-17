"""Immutable inference namespaces; no destructive global tmp cleanup."""
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(value, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_cache(opt, dataset, engine):
    root = Path(getattr(opt, '_inference_root', opt.inference_dir)).resolve()
    opt._inference_root = str(root)
    weights = {name: sha256_file(getattr(opt, name)) for name in ('rgb_model', 'flow_model') if getattr(opt, name)}
    if not weights:
        raise ValueError('Inference requires --rgb_model or --flow_model')
    text_revision = None
    if opt.modality == 'multimodal':
        text_root = Path(opt.text_root or Path(dataset.ROOT_DATASET_PATH) / 'numpys')
        manifest = text_root / 'features.json'
        if manifest.is_file():
            text_revision = sha256_file(manifest)
        elif opt.text_revision:
            text_revision = opt.text_revision
        else:
            raise ValueError('Multimodal cache requires exported features.json or explicit --text_revision')
    source_root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        source.update(str(path.relative_to(source_root)).encode())
        source.update(path.read_bytes())
    config = {key: getattr(opt, key) for key in (
        'dataset', 'split', 'eval_split', 'arch', 'num_classes', 'K', 'ninput',
        'resize_height', 'resize_width', 'down_ratio', 'flip_test', 'N', 'modality',
        'text_format', 'hm_fusion_rgb', 'mov_fusion_rgb', 'wh_fusion_rgb', 'dcn_backend')}
    config.update(weights=weights, source=source.hexdigest(), engine=engine,
                  data_root=str(Path(dataset.ROOT_DATASET_PATH).resolve()),
                  gt_sha256=sha256_file(dataset.annotation_file),
                  text_root=str(Path(opt.text_root).resolve()) if opt.text_root else None,
                  text_revision=text_revision, videos=dataset.video_list)
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    directory = root / hashlib.sha256(encoded).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    # Identical writers have identical manifest bytes; exclusive creation avoids truncation.
    try:
        with (directory / 'inference.json').open('xb') as stream:
            stream.write(encoded)
    except FileExistsError:
        if (directory / 'inference.json').read_bytes() != encoded:
            raise ValueError('Inference manifest mismatch: {}'.format(directory))
    opt.inference_dir = str(directory)
    print('Inference output: {}'.format(directory))


def require_cache_protocol(opt, dataset):
    path = Path(opt.inference_dir) / 'inference.json'
    if not path.is_file():
        if opt.allow_legacy_predictions:
            import warnings
            warnings.warn('Evaluating legacy predictions without provenance; protocol is unverified')
            return
        raise ValueError('Use the inference output directory printed by det.py, or explicitly --allow_legacy_predictions')
    saved = json.loads(path.read_text())
    expected = {key: getattr(opt, key) for key in ('dataset', 'split', 'eval_split', 'K', 'num_classes', 'modality', 'text_format')}
    expected.update(videos=dataset.video_list, gt_sha256=sha256_file(dataset.annotation_file))
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Evaluation protocol differs from cached predictions')
