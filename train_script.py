import argparse
import os.path
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.datamodules.super_datamodule import SuperDataModule
from thesislib.models import CLIPIDP

from thesislib.datamodules import \
    CIFAR100DataModule, DTDDataModule, SUN397DataModule, Food101DataModule, \
    Flowers102DataModule, EuroSATDataModule, UCF101DataModule, \
    OxfordPetsDataModule, CIFAR10DataModule, SVHNDataModule, RESISC45DataModule, \
    CLEVRCountDataModule

datamodules = {
    'cifar10': CIFAR10DataModule,
    'cifar100': CIFAR100DataModule,
    'dtd': DTDDataModule,
    'sun397': SUN397DataModule,
    'food101': Food101DataModule,
    'flowers102': Flowers102DataModule,
    'eurosat': EuroSATDataModule,
    'ucf101': UCF101DataModule,
    'oxford_pets': OxfordPetsDataModule,
    'svhn': SVHNDataModule,
    'resisc45': RESISC45DataModule,
    'clevr_count': CLEVRCountDataModule,
    'super': SuperDataModule,
}

visual_embedding_dims = {
    'ViT-B/32': 768,
    'ViT-B/16': 768,
    'ViT-L/14': 1024,
}
text_embedding_dims = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 768,
}


def main(args):
    seed_everything(seed=args.seed, workers=True)

    datamodule = datamodules[args.dataset](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scale_lower_bound=args.rrc_scale_lb,
        jitter_prob=args.jitter_prob,
        greyscale_prob=args.greyscale_prob,
        solarize_prob=args.solarize_prob,
    )

    if not args.disable_idp:
        idp_settings = {
            'idp_mode': args.idp_mode,
            'idp_resolution': args.idp_resolution,
            'nr_output_vectors': args.idp_length,
            'vector_dim': visual_embedding_dims[args.architecture],
            'mixture_size': args.idp_mixture_size,
            'pretrained_idp': args.pretrained_idp,
            'hybrid_idp_mode': args.hybrid_idp_mode,
            'model_type': args.idp_model_type,
            'proj_type': args.idp_proj_type,
            'idp_act_fn': args.idp_act_fn,
            'nr_groups': args.nr_groups,
            'blocks_per_group': args.blocks_per_group,
            'initial_channels': args.initial_channels,
            'init_max_pool': args.init_max_pool

        }
    else:
        idp_settings = None

    clip_idp = CLIPIDP(
        clip_architecture=args.architecture,
        idp_settings=idp_settings,
        optimizer=args.optimizer,
        init_lr=args.init_lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        entropy_loss_coeff=args.entropy_loss_coeff,
        disable_loggers=args.disable_loggers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{step}-{val_loss:.2f}',
        save_last=True,
        mode='min',
        auto_insert_metric_name=True,
        save_top_k=5)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=args.epochs,
        precision=args.precision,
        track_grad_norm=2,
        strategy=args.strategy,
        profiler='simple' if args.profiler else None,
        fast_dev_run=30 if args.dev_run else False,
    )

    trainer.logger.log_hyperparams(args)

    trainer.fit(
        model=clip_idp,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='super', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)
    parser.add_argument('--rrc_scale_lb', default=1., type=float)
    parser.add_argument('--jitter_prob', default=0.8, type=float)
    parser.add_argument('--greyscale_prob', default=0.2, type=float)
    parser.add_argument('--solarize_prob', default=0.2, type=float)

    # Model + Training
    parser.add_argument('--architecture', default='ViT-B/32', type=str)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--entropy_loss_coeff', default=0, type=float)

    # IDP
    parser.add_argument('--idp_length', default=16, type=int)
    parser.add_argument('--idp_mode', default='hybrid', type=str)
    parser.add_argument('--idp_mixture_size', default=128, type=int)
    parser.add_argument('--idp_act_fn', default='softmax', type=str)
    parser.add_argument('--hybrid_idp_mode', default='shared', type=str)

    # IDP Model
    parser.add_argument('--idp_model_type', default='small', type=str)
    parser.add_argument('--idp_proj_type', default='linear', type=str)
    parser.add_argument('--idp_resolution', default=224, type=int)
    parser.add_argument('--nr_groups', default=4, type=int)
    parser.add_argument('--blocks_per_group', default=1, type=int)
    parser.add_argument('--initial_channels', default=16, type=int)
    parser.add_argument('--init_max_pool', action=argparse.BooleanOptionalAction,
                        default=True)

    # Optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--init_lr', default=0.1, type=float)
    parser.add_argument('--lr_scheduler', default='warmup', type=str)

    # Experiment
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Switches
    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_idp', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--pretrained_idp', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_loggers',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    main(args)
