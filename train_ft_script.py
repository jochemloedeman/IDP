import argparse
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent / "thesislib"))
from thesislib.models.finetune_clip import FinetuneClip

from thesislib.datamodules.imagenet_a_datamodule import ImageNetADataModule
from thesislib.datamodules.imagenet_datamodule import ImageNetDataModule
from thesislib.datamodules.imagenet_r_datamodule import ImageNetRDataModule
from thesislib.datamodules.imagenet_sketch_datamodule import ImageNetSketchDataModule
from thesislib.datamodules.imagenet_v2_datamodule import ImageNetV2DataModule

from thesislib.datamodules import (
    CIFAR100DataModule,
    DTDDataModule,
    SUN397DataModule,
    Food101DataModule,
    Flowers102DataModule,
    EuroSATDataModule,
    UCF101DataModule,
    OxfordPetsDataModule,
    CIFAR10DataModule,
    SVHNDataModule,
    RESISC45DataModule,
    CLEVRCountDataModule,
    # MegaDataModule,
)

datamodules = {
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
    "dtd": DTDDataModule,
    "sun397": SUN397DataModule,
    "food101": Food101DataModule,
    "flowers102": Flowers102DataModule,
    "eurosat": EuroSATDataModule,
    "ucf101": UCF101DataModule,
    "oxford_pets": OxfordPetsDataModule,
    "svhn": SVHNDataModule,
    "resisc45": RESISC45DataModule,
    "clevr_count": CLEVRCountDataModule,
    "imagenet": ImageNetDataModule,
    "imagenet_a": ImageNetADataModule,
    "imagenet_r": ImageNetRDataModule,
    "imagenet_v2": ImageNetV2DataModule,
    "imagenet_sketch": ImageNetSketchDataModule,
}

visual_embedding_dims = {
    "ViT-B/32": 768,
    "ViT-B/16": 768,
    "ViT-L/14": 1024,
}
text_embedding_dims = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
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

    clip_idp = FinetuneClip(
        clip_architecture=args.architecture,
        nr_classes=datamodule.nr_of_classes,
        optimizer=args.optimizer,
        init_lr=args.init_lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        full_finetune=args.full_finetune
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min",
        auto_insert_metric_name=True,
        save_top_k=5,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=args.epochs,
        precision=args.precision,
        track_grad_norm=2,
        strategy=args.strategy,
        profiler="simple" if args.profiler else None,
        fast_dev_run=30 if args.dev_run else False,
    )

    trainer.logger.log_hyperparams(args)

    if args.ckpt_file_name:
        ckpt_path = Path(__file__).parent / "checkpoints" / args.ckpt_file_name
    else:
        ckpt_path = None

    trainer.fit(model=clip_idp, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--data_root", default="/home/jochem/Documents/data", type=str
    )
    parser.add_argument("--rrc_scale_lb", default=1.0, type=float)
    parser.add_argument("--jitter_prob", default=0.8, type=float)
    parser.add_argument("--greyscale_prob", default=0.2, type=float)
    parser.add_argument("--solarize_prob", default=0.2, type=float)
    # Model + Training
    parser.add_argument("--architecture", default="ViT-B/32", type=str)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--ckpt_file_name", default="", type=str)

    # Optimizer
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--init_lr", default=0.1, type=float)
    parser.add_argument("--lr_scheduler", default="warmup", type=str)

    # Experiment
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--strategy", default=None, type=str)
    parser.add_argument("--devices", default=-1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Switches
    parser.add_argument(
        "--dev_run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--profiler", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--full_finetune", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    main(args)
