import argparse
import sys
from pathlib import Path

import torch.nn
from pytorch_lightning import seed_everything
from torchray.utils import get_device

from thesislib.attribution.generate import generate_perturbation_mask, \
    generate_gradcam_map, generate_item_specific_attributions, \
    generate_image_specific_attributions
from thesislib.components.idp import HybridSharedIDP

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
        scale_lower_bound=args.rrc_scale_lb,
        jitter_prob=args.jitter_prob,
        greyscale_prob=args.greyscale_prob,
        solarize_prob=args.solarize_prob,
    )
    datamodule.setup(stage='test')

    if args.ckpt_file_name:
        clip_idp = CLIPIDP.load_from_checkpoint(
            Path(__file__).parent / 'checkpoints' / args.ckpt_file_name
        )
        print(f"loaded {args.ckpt_file_name}\n")
        idp_module = wrap_lightning_module(clip_idp.input_dependent_prompt,
                                       args.token_idx)
    else:
        idp_module = HybridSharedIDP
    idp_module.to(device)
    test_set = datamodule.test_set

    if args.attr_vis_mode == 'item_specific':
        generate_item_specific_attributions(
            test_set,
            device,
            args.attr_method,
            idp_module,
            args.dict_idx,
            args.nr_rows_cols,
            args.disable_mask,
        )
    elif args.attr_vis_mode == 'image_specific':
        generate_image_specific_attributions(
            test_set,
            device,
            args.attr_method,
            idp_module,
            args.image_idx,
            args.nr_rows_cols,
            args.disable_mask,
        )


def wrap_lightning_module(lightning_module, token_idx):
    class ModuleWrapper(torch.nn.Module):
        def __init__(self, idp_module, token_idx):
            super().__init__()
            self.idp_module = idp_module
            self.token_idx = token_idx

        def forward(self, x):
            idps, mixture_logits = self.idp_module(x)
            return mixture_logits[:, self.token_idx]

    return ModuleWrapper(lightning_module, token_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='sun397', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)

    parser.add_argument('--ckpt_file_name', default="8x8_sun397.ckpt", type=str)
    parser.add_argument('--rrc_scale_lb', default=0.875, type=float)
    parser.add_argument('--jitter_prob', default=0.0, type=float)
    parser.add_argument('--greyscale_prob', default=0.0, type=float)
    parser.add_argument('--solarize_prob', default=0.0, type=float)

    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--token_idx', default=0, type=int)
    parser.add_argument('--dict_idx', default=5, type=int)
    parser.add_argument('--image_idx', default=0, type=int)
    parser.add_argument('--nr_rows_cols', default=3, type=int)
    parser.add_argument('--attr_method', default='grad-cam', type=str)
    parser.add_argument('--attr_vis_mode', default='item_specific', type=str)
    parser.add_argument('--disable_mask',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    main(args)
