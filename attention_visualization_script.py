import argparse
import random
import sys
from pathlib import Path

import torch.nn
import torchvision
from pytorch_lightning import seed_everything
from torchray.utils import get_device

from thesislib.probe_clip import AttentionProbe, generate_attention_maps, \
    visualize_with_idp
from thesislib.probe_clip.extract_statistics import extract_statistics
from thesislib.probe_clip.visualize import visualize_comparison

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.models import CLIPIDP
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
    'ViT-L/14': 512,
}


def main(args):
    # seed_everything(seed=args.seed, workers=True)

    device = get_device()

    sun397_datamodule = datamodules['sun397'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    sun397_datamodule.setup(stage='test')

    cifar100_datamodule = datamodules['cifar100'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    cifar100_datamodule.setup(stage='test')

    ucf101_datamodule = datamodules['ucf101'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    ucf101_datamodule.setup(stage='test')

    resisc45_datamodule = datamodules['resisc45'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    resisc45_datamodule.setup(stage='test')

    oxford_pets_datamodule = datamodules['oxford_pets'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    oxford_pets_datamodule.setup(stage='test')

    food101_datamodule = datamodules['food101'](
        data_root=args.data_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        scenario=args.scenario,
        scale_lower_bound=1,
        jitter_prob=0,
        greyscale_prob=0,
        solarize_prob=0,
    )
    food101_datamodule.setup(stage='test')

    sun397_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_sun397.ckpt"
    )
    cifar100_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_cifar100.ckpt"
    )
    ucf101_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_ucf101.ckpt"
    )
    food101_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_food101.ckpt"
    )
    oxford_pets_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_oxford_pets.ckpt"
    )
    resisc45_clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / "16x128_resisc45.ckpt"
    )

    c = [random.randint(0, len(food101_datamodule.test_set))]
    print(c)
    sun397_indices = [100]
    cifar100_indices = [6890]
    ucf101_indices = [15]
    food101_indices = c
    oxford_pets_indices = [145]
    resisc45_indices = [220]

    sun397_attention_probe = AttentionProbe(sun397_clip_idp)
    cifar100_attention_probe = AttentionProbe(cifar100_clip_idp)
    ucf101_attention_probe = AttentionProbe(ucf101_clip_idp)
    food101_attention_probe = AttentionProbe(food101_clip_idp)
    oxford_pets_attention_probe = AttentionProbe(oxford_pets_clip_idp)
    resisc45_attention_probe = AttentionProbe(resisc45_clip_idp)

    with torch.no_grad():
        sun397_attention_maps_pgn = generate_attention_maps(
            attention_probe=sun397_attention_probe,
            test_set=sun397_datamodule.test_set,
            example_indices=sun397_indices,
            device=device
        )
        del sun397_attention_probe
        del sun397_clip_idp
        cifar100_attention_maps_pgn = generate_attention_maps(
            attention_probe=cifar100_attention_probe,
            test_set=cifar100_datamodule.test_set,
            example_indices=cifar100_indices,
            device=device
        )
        del cifar100_attention_probe
        del cifar100_clip_idp
        ucf101_attention_maps_pgn = generate_attention_maps(
            attention_probe=ucf101_attention_probe,
            test_set=ucf101_datamodule.test_set,
            example_indices=ucf101_indices,
            device=device
        )
        del ucf101_attention_probe
        del ucf101_clip_idp
        food101_attention_maps_pgn = generate_attention_maps(
            attention_probe=food101_attention_probe,
            test_set=food101_datamodule.test_set,
            example_indices=food101_indices,
            device=device
        )
        del food101_attention_probe
        del food101_clip_idp
        oxford_pets_attention_maps_pgn = generate_attention_maps(
            attention_probe=oxford_pets_attention_probe,
            test_set=oxford_pets_datamodule.test_set,
            example_indices=oxford_pets_indices,
            device=device
        )
        del oxford_pets_attention_probe
        del oxford_pets_clip_idp
        resisc45_attention_maps_pgn = generate_attention_maps(
            attention_probe=resisc45_attention_probe,
            test_set=resisc45_datamodule.test_set,
            example_indices=resisc45_indices,
            device=device
        )
        del resisc45_attention_probe
        del resisc45_clip_idp

        clip_idp = CLIPIDP()
        attention_probe = AttentionProbe(clip_idp)

        sun397_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=sun397_datamodule.test_set,
            example_indices=sun397_indices,
            device=device
        )
        cifar100_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=cifar100_datamodule.test_set,
            example_indices=cifar100_indices,
            device=device
        )
        ucf101_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=ucf101_datamodule.test_set,
            example_indices=ucf101_indices,
            device=device
        )
        food101_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=food101_datamodule.test_set,
            example_indices=food101_indices,
            device=device
        )
        oxford_pets_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=oxford_pets_datamodule.test_set,
            example_indices=oxford_pets_indices,
            device=device
        )
        resisc45_attention_maps_clip = generate_attention_maps(
            attention_probe=attention_probe,
            test_set=resisc45_datamodule.test_set,
            example_indices=resisc45_indices,
            device=device
        )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
    ])

    sun397_datamodule.test_set.transforms = test_transform
    cifar100_datamodule.test_set.transform = test_transform
    ucf101_datamodule.test_set.transforms = test_transform
    food101_datamodule.test_set.transforms = test_transform
    oxford_pets_datamodule.test_set.transforms = test_transform
    resisc45_datamodule.test_set.transforms = test_transform

    imgs = [sun397_datamodule.test_set[index][0] for index in
            sun397_indices]
    imgs += [cifar100_datamodule.test_set[index][0] for index in
             cifar100_indices]
    imgs += [ucf101_datamodule.test_set[index][0] for index in
             ucf101_indices]
    imgs += [food101_datamodule.test_set[index][0] for index in
             food101_indices]
    imgs += [oxford_pets_datamodule.test_set[index][0] for index in
             oxford_pets_indices]
    imgs += [resisc45_datamodule.test_set[index][0] for index in
             resisc45_indices]

    visualize_comparison(
        torch.cat(
            [sun397_attention_maps_clip,
             cifar100_attention_maps_clip,
             ucf101_attention_maps_clip,
             food101_attention_maps_clip,
             oxford_pets_attention_maps_clip,
             resisc45_attention_maps_clip]
        ),
        torch.cat(
            [sun397_attention_maps_pgn,
             cifar100_attention_maps_pgn,
             ucf101_attention_maps_pgn,
             food101_attention_maps_pgn,
             oxford_pets_attention_maps_pgn,
             resisc45_attention_maps_pgn]
        ),
        imgs,
        args.token_idx,
        args.idp_length if args.ckpt_file_name else 0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='sun397', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)

    parser.add_argument('--ckpt_file_name', default="8x64_sun397_resnet10.ckpt",
                        type=str)

    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--idp_length', default=16, type=int)
    parser.add_argument('--example_idx', default=100, type=int)
    parser.add_argument('--token_idx', default=0, type=int)

    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    main(args)
