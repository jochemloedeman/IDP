import argparse
import os
import sys
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.datamodules import CIFAR100DataModule

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.models import CLIPIDP

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
    'ViT-L/14': 512,
}


def main(args):
    seed_everything(seed=args.seed, workers=True)

    zeroshot_file = os.path.join(args.data_root,
                                 args.dataset,
                                 "zeroshot_classes.txt")

    zeroshot_classes = [int(elem) for elem
                        in open(zeroshot_file, 'r').read().splitlines()]

    datamodule = datamodules[args.dataset](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario='regular'
    )



    if args.ckpt_file_name:
        clip_idp = CLIPIDP.load_from_checkpoint(
            Path(__file__).parent / 'checkpoints' / args.ckpt_file_name
        )
        print(f"loaded {args.ckpt_file_name}\n")
    else:
        clip_idp = CLIPIDP(
            clip_architecture=args.architecture,
            optimizer=args.optimizer,
            unseen_classes=zeroshot_classes
            if args.scenario == 'zeroshot' else None,
            idp_settings=None,
        )
        print("Loaded 0-shot CLIP\n")

    test_trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=args.precision,
    )

    test_trainer.test(
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
    parser.add_argument('--ckpt_file_name',
                        default="",
                        type=str)
    parser.add_argument('--architecture', default='ViT-B/32', type=str)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--idp_length', default=6, type=int)
    parser.add_argument('--idp_mode', default='hybrid', type=str)
    parser.add_argument('--idp_mixture_size', default=3, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--strategy', default='ddp', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dev_run', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--profiler', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--pretrained_idp',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    main(args)

