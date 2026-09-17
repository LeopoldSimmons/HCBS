"""Training with explicit modalities, validation partitions and full resume."""
import copy
import os
import random

import numpy as np
import tensorboardX
import torch
from opts import opts
from MOC_utils.model import (create_model, convert2flow, load_model, save_model,
                             load_coco_pretrained_model, load_imagenet_pretrained_model)
from MOC_utils.checkpoint import read_checkpoint, check_metadata, restore_rng, text_revision
from trainer.logger import Logger
from datasets.init_dataset import get_dataset
from trainer.moc_trainer import MOCTrainer
from inference.normal_inference import normal_inference
from ACT import frameAP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    if opt.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.deterministic
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.use_deterministic_algorithms(opt.deterministic)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)
    text_revision(opt)
    train_dataset = Dataset(opt, 'train')
    val_dataset = Dataset(opt, 'val') if opt.val_epoch or opt.auto_stop else None
    model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K)
    if opt.load_model:
        if opt.ninput > 1:
            model = convert2flow(opt.ninput, model)
        if not opt.ucf_pretrain:
            check_metadata(read_checkpoint(opt.load_model, training=opt.resume), opt)
        model = load_model(model, opt.load_model, ucf_pretrain=opt.ucf_pretrain)
    elif opt.pretrain_model == 'coco':
        model = load_coco_pretrained_model(opt, model)
    else:
        model = load_imagenet_pretrained_model(opt, model)
    # Structure and input-channel conversions must finish before optimizer creation.
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch, best_ap, best_epoch = opt.start_epoch, float('-inf'), None
    resume_rng = None
    if opt.resume:
        checkpoint = read_checkpoint(opt.load_model, training=True)
        if 'rng' not in checkpoint or 'training_state' not in checkpoint:
            raise ValueError('Full resume requires RNG and training_state; load legacy weights without --resume')
        saved = checkpoint['training_state']
        if saved.get('lr_step') != opt.lr_step or saved.get('base_lr') != opt.lr:
            raise ValueError('Resume LR schedule differs from checkpoint')
        model, optimizer, start_epoch, best_ap = load_model(model, opt.load_model, optimizer)
        best_epoch = saved.get('best_epoch')
        resume_rng = checkpoint['rng']
    elif start_epoch:
        raise ValueError('Use --resume to restore epoch/LR; --start_epoch alone cannot recover training state')
    trainer = MOCTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    loader_args = dict(batch_size=opt.batch_size, num_workers=opt.num_workers,
                       pin_memory=opt.pin_memory, worker_init_fn=worker_init_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    if not len(train_loader):
        raise ValueError('No training batches; reduce --batch_size or check partition and K')
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args) if opt.val_epoch else None
    if val_loader is not None and not len(val_loader):
        raise ValueError('No validation tubelets; check val partition and K')
    os.makedirs(opt.save_dir, exist_ok=True)
    writers = [tensorboardX.SummaryWriter(os.path.join(opt.log_dir, name))
               for name in ('train', 'train_epoch', 'val', 'val_epoch')]
    train_writer, epoch_train_writer, val_writer, epoch_val_writer = writers
    logger = Logger(opt, epoch_train_writer, epoch_val_writer)
    if resume_rng is not None:
        restore_rng(resume_rng)
    try:
        for epoch in range(start_epoch + 1, opt.num_epochs + 1):
            train_stats = trainer.train(epoch, train_loader, train_writer)
            for key, value in train_stats.items():
                logger.scalar_summary('epoch/' + key, value, epoch, 'train')
            if opt.val_epoch:
                with torch.no_grad():
                    val_stats = trainer.val(epoch, val_loader, val_writer)
                for key, value in val_stats.items():
                    logger.scalar_summary('epoch/' + key, value, epoch, 'val')
            improved = False
            if opt.auto_stop:
                evaluation_model = os.path.join(opt.save_dir, 'model_eval.pth')
                save_model(evaluation_model, model, epoch=epoch, opt=opt)
                evaluation = copy.copy(opt)
                evaluation.eval_split = 'val'
                evaluation.inference_dir = os.path.join(opt.save_dir, 'validation_predictions')
                if hasattr(evaluation, '_inference_root'):
                    del evaluation._inference_root
                evaluation.rgb_model = evaluation_model if opt.ninput == 1 else ''
                evaluation.flow_model = evaluation_model if opt.ninput > 1 else ''
                # Identical normal inference path is used for visual and multimodal selection.
                normal_inference(evaluation)
                ap = frameAP(evaluation, print_info=opt.print_log)
                if not np.isfinite(ap):
                    raise ValueError('Non-finite validation AP at epoch {}'.format(epoch))
                improved = best_epoch is None or ap > best_ap
                if improved:
                    best_ap, best_epoch = float(ap), epoch
                logger.scalar_summary('epoch/frameAP', ap, epoch, 'val')
            # A conventional epoch schedule; best weights do not reset optimizer history.
            next_lr = opt.lr * (0.1 ** sum(epoch >= step for step in opt.lr_step))
            for group in optimizer.param_groups:
                group['lr'] = next_lr
            state = {'best_epoch': best_epoch, 'base_lr': opt.lr, 'lr_step': opt.lr_step}
            save_args = dict(optimizer=optimizer, epoch=epoch, best=best_ap, opt=opt, training_state=state)
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), model, **save_args)
            if improved:
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), model, **save_args)
            if opt.save_all:
                save_model(os.path.join(opt.save_dir, 'model_{:04d}.pth'.format(epoch)), model, **save_args)
            logger.write('epoch {} train={} best_val_ap={} best_epoch={}\n'.format(epoch, train_stats, best_ap, best_epoch))
    finally:
        logger.close()
        for writer in writers:
            writer.close()


if __name__ == '__main__':
    main(opts().parse())
