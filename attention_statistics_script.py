import argparse
import random
import sys
from pathlib import Path

import torch.nn
from pytorch_lightning import seed_everything
from torchray.utils import get_device

from thesislib.probe_clip import AttentionProbe, generate_attention_maps, \
    visualize_with_idp
from thesislib.probe_clip.extract_statistics import extract_statistics

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.models import CLIPIDP
from thesislib.datamodules import CIFAR100DataModule, DTDDataModule, \
    SUN397DataModule, Food101DataModule

datamodules = {
    'cifar100': CIFAR100DataModule,
    'dtd': DTDDataModule,
    'sun397': SUN397DataModule,
    'food101': Food101DataModule,
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

    device = get_device()

    datamodule = datamodules[args.dataset](
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
    datamodule.setup(stage='test')

    if args.ckpt_file_name:
        clip_idp = CLIPIDP.load_from_checkpoint(
            Path(__file__).parent / 'checkpoints' / args.ckpt_file_name
        )
        print(f"loaded {args.ckpt_file_name}\n")
    else:
        clip_idp = CLIPIDP()

    attention_probe = AttentionProbe(clip_idp)
    len_test_set = len(datamodule.test_set)
    image_indices = random.choices(list(range(len_test_set)), k=1000)
    attention_maps = generate_attention_maps(
        attention_probe=attention_probe,
        test_set=datamodule.test_set,
        example_indices=image_indices,
        device=device
    )
    seq_length = attention_maps.shape[-1]
    if args.ckpt_file_name:
        indices = {
            'cls': torch.LongTensor([0]),
            'patch': torch.LongTensor(range(1, seq_length-args.idp_length)),
            'idp': torch.LongTensor(range(seq_length-args.idp_length, seq_length))
        }
    else:
        indices = {
            'cls': torch.LongTensor([0]),
            'patch': torch.LongTensor(range(1, seq_length)),
        }

    statistics = extract_statistics(attention_maps, indices['patch'], indices['idp'])
    print(statistics.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='sun397', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)

    parser.add_argument('--ckpt_file_name', default="16x128_sun397.ckpt", type=str)

    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--idp_length', default=16, type=int)
    parser.add_argument('--token_idx', default=0, type=int)

    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    main(args)
