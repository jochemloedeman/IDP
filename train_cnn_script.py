import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))

from thesislib.models import CNN

from thesislib.datamodules import \
    CIFAR100DataModule, DTDDataModule, SUN397DataModule, Food101DataModule, \
    Flowers102DataModule, EuroSATDataModule, UCF101DataModule, \
    OxfordPetsDataModule, CIFAR10DataModule, SVHNDataModule, RESISC45DataModule, \
    CLEVRCountDataModule, MegaDataModule, SuperDataModule

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
    'mega': MegaDataModule,
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

    cnn = CNN(
        nr_of_classes=datamodule.nr_of_classes,
        cnn_settings=vars(args),
        optimizer=args.optimizer,
        init_lr=args.init_lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        pretrained=args.pretrained
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
    )

    trainer.logger.log_hyperparams(args)

    trainer.fit(
        model=cnn,
        datamodule=datamodule,
    )
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)
    parser.add_argument('--rrc_scale_lb', default=0.875, type=float)
    parser.add_argument('--jitter_prob', default=0.0, type=float)
    parser.add_argument('--greyscale_prob', default=0.0, type=float)
    parser.add_argument('--solarize_prob', default=0.0, type=float)

    # Model + Training
    parser.add_argument('--pretrained',
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--val_batch_size', default=256, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--nr_groups', default=4, type=int)
    parser.add_argument('--blocks_per_group', default=1, type=int)
    parser.add_argument('--initial_channels', default=16, type=int)
    parser.add_argument('--init_max_pool',
                        action=argparse.BooleanOptionalAction,
                        default=True)

    # Optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--init_lr', default=0.1, type=float)
    parser.add_argument('--lr_scheduler', default='warmup', type=str)

    # Experiment
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)


    args = parser.parse_args()

    main(args)
