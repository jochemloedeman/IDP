import argparse
import sys
from pathlib import Path

import torch.nn
from pytorch_lightning import seed_everything
from torchray.utils import get_device

sys.path.append(str(Path(__file__).parent.parent / 'thesislib'))
from thesislib.attribution.generate import generate_item_specific_attributions, \
    find_top_dict_items, create_image_grids
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
    print(device)

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

    clip_idp = CLIPIDP.load_from_checkpoint(
        Path(__file__).parent / 'checkpoints' / args.ckpt_file_name
    )
    print(f"loaded {args.ckpt_file_name}\n")

    test_set = datamodule.test_set
    for token_index in range(8):
        idp_module = wrap_lightning_module(clip_idp.input_dependent_prompt,
                                           token_index)
        idp_module.to(device)
        idp_module.eval()
        dict_items = find_top_dict_items(idp_module, test_set, device)
        dict_items = dict_items.cpu().tolist()
        for dict_item in dict_items[:2]:
            images = generate_item_specific_attributions(
                test_set,
                device,
                args.attr_method,
                idp_module,
                dict_item,
                args.nr_rows_cols,
                args.disable_mask,
            )

            create_image_grids(images,
                               args.nr_rows_cols,
                               args.dataset,
                               args.ckpt_file_name,
                               token_index,
                               dict_item)
        torch.cuda.empty_cache()


def wrap_lightning_module(lightning_module, token_idx):
    class ModuleWrapper(torch.nn.Module):
        def __init__(self, idp_module, token_idx):
            super().__init__()
            self.idp_module = idp_module
            self.idp_module.log = False
            self.token_idx = token_idx

        def forward(self, x):
            idps, mixture_logits = self.idp_module(x)
            token_logits = mixture_logits[:, self.token_idx]
            del idps
            del mixture_logits
            torch.cuda.empty_cache()
            return token_logits

    return ModuleWrapper(lightning_module, token_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--data_root',
                        default='/home/jochem/Documents/ai/scriptie/data',
                        type=str)

    parser.add_argument('--ckpt_file_name',
                        default="8x64_cifar100_resnet10.ckpt",
                        type=str)
    parser.add_argument('--rrc_scale_lb', default=0.875, type=float)
    parser.add_argument('--jitter_prob', default=0.0, type=float)
    parser.add_argument('--greyscale_prob', default=0.0, type=float)
    parser.add_argument('--solarize_prob', default=0.0, type=float)

    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--nr_rows_cols', default=4, type=int)
    parser.add_argument('--attr_method', default='perturbation', type=str)
    parser.add_argument('--attr_vis_mode', default='item_specific', type=str)
    parser.add_argument('--disable_mask',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    main(args)
