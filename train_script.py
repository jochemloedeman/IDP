import argparse
import os.path
import sys
import pytorch_lightning as pl
from pathlib import Path

from pytorch_lightning import seed_everything
sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.models import CLIPIDP
from thesislib.datamodules import CIFAR100DataModule


datamodules = {
    'cifar100': CIFAR100DataModule,
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
        scenario=args.scenario
    )

    if not args.disable_idp:
        idp_settings = {
            'idp_mode': args.idp_mode,
            'nr_output_vectors': args.idp_length,
            'vector_dim': visual_embedding_dims[args.architecture],
            'mixture_size': args.idp_mixture_size,
            'pretrained_idp': args.pretrained_idp,
        }
    else:
        idp_settings = None

    zeroshot_file = os.path.join(args.data_root,
                                 args.dataset,
                                 "zeroshot_classes.txt")

    zeroshot_classes = [int(elem) for elem
                        in open(zeroshot_file, 'r').read().splitlines()]

    clip_idp = CLIPIDP(
        clip_architecture=args.architecture,
        idp_settings=idp_settings,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        epochs=args.epochs,
        unseen_classes=zeroshot_classes if args.scenario == 'zeroshot' else None
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
    parser.add_argument('--scenario', default='zeroshot', type=str)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)
    parser.add_argument('--architecture', default='ViT-B/32', type=str)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--idp_length', default=8, type=int)
    parser.add_argument('--idp_mode', default='constant', type=str)
    parser.add_argument('--idp_mixture_size', default=20, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--strategy', default='ddp', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--disable_idp', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--pretrained_idp', action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    main(args)
