import argparse
import os.path

from matplotlib import pyplot as plt, gridspec


def main(args):
    cifar100_image_paths = [
        "0/13_#1.png",
        "2/1_#3.png",
        "3/4_#0.png"
    ]
    sun397_image_paths = [
        "0/30_#2.png",
        "1/39_#1.png",
        "2/7_#0.png"
    ]
    ucf101_image_paths = [
        "0/6_#0.png",
        "0/129_#2.png",
        "1/60_#0.png",
    ]

    cifar100_model = "8x64_cifar100_resnet10.ckpt"
    sun397_model = "8x64_sun397_resnet10.ckpt"
    ucf101_model = "16x128_ucf101.ckpt"
    attribution_path = "/home/jochem/Documents/ai/scriptie/idp/attribution"
    cifar100_path = os.path.join(attribution_path, 'cifar100', cifar100_model)
    sun397_path = os.path.join(attribution_path, 'sun397', sun397_model)
    ucf101_path = os.path.join(attribution_path, 'ucf101', ucf101_model)

    cifar100_images = []
    for image_path in cifar100_image_paths:
        cifar100_images += [plt.imread(os.path.join(cifar100_path, image_path))]

    sun397_images = []
    for image_path in sun397_image_paths:
        sun397_images += [plt.imread(os.path.join(sun397_path, image_path))]

    ucf101_images = []
    for image_path in ucf101_image_paths:
        ucf101_images += [plt.imread(os.path.join(ucf101_path, image_path))]

    fig = plt.figure(figsize=(17, 4), dpi=300)
    outer_grid = gridspec.GridSpec(1, 3, hspace=0.0, wspace=0.01)

    visualize_dataset(cifar100_images, outer_grid[0], "CIFAR100")
    visualize_dataset(sun397_images, outer_grid[1], "SUN397")
    visualize_dataset(ucf101_images, outer_grid[2], "UCF101")
    plt.tight_layout()
    plt.savefig(f"attribution/attribution_compilation.png",
                dpi=300, bbox_inches='tight')


def visualize_dataset(row_images, grid, title):
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        nrows=len(row_images),
        ncols=1,
        subplot_spec=grid,
        hspace=0.0,
    )
    ax = plt.subplot(inner_grid[0])
    ax.set_title(title)
    for idx in range(len(row_images)):
        visualize_attr_row(row_images[idx], inner_grid[idx])


def visualize_attr_row(row_image, grid):
    ax = plt.subplot(grid)
    ax.tick_params(left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.imshow(row_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--scenario', default='regular', type=str)
    parser.add_argument('--dataset', default='sun397', type=str)

    args = parser.parse_args()

    main(args)
