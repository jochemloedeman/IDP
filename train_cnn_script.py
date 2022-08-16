import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.models import CNN
from thesislib.datamodules import \
    CIFAR100DataModule, DTDDataModule, SUN397DataModule, Food101DataModule

datamodules = {
    'cifar100': CIFAR100DataModule,
    'dtd': DTDDataModule,
    'sun397': SUN397DataModule,
    'food101': Food101DataModule,
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

    cnn = CNN(
        model_type=args.model_type,
        nr_of_classes=datamodule.nr_of_classes,
        optimizer=args.optimizer,
        init_lr=args.init_lr,
        lr_scheduler=args.lr_scheduler,
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
        profiler='simple' if args.profiler else None,
        fast_dev_run=30 if args.dev_run else False,
    )

    trainer.logger.log_hyperparams(args)

    trainer.fit(
        model=cnn,
        datamodule=datamodule,
    )
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='dtd', type=str)
    parser.add_argument('--model_type', default='small', type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--init_lr', default=1e-2, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    main(args)
