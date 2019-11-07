from model import Generator
from os.path import join
import argparse
import os
import torch
import helper
import matplotlib.pyplot as plt
import yaml


def output_fig(images_array, file_name):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name + ".jpg", bbox_inches="tight", pad_inches=0)
    plt.close()


def test(args):
    try:
        os.makedirs(args.save_img_path)
    except OSError:
        pass

    # model
    generator = Generator(args.bs, args.imgsize, z_dim=args.z_size).cuda()
    generator.load_state_dict(
        torch.load(join(f"{args.weight_path}", f"generator_{args.epochs}.pth"))
    )

    # test
    generator.eval()
    with torch.no_grad():
        for i in range(100):
            z = torch.randn(args.bs, args.z_size)
            z = torch.tensor(z).cuda()
            generated_images, _, _ = generator(z)
            generated_images = generated_images.permute(0, 2, 3, 1)
            generated_images = generated_images.data.cpu().numpy()
            output_fig(
                generated_images,
                file_name="{}/{}_image".format(
                    args.save_img_path, str.zfill(str(i + 1), 3)
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["test"].items():
        setattr(args, key, value)
    test(args)
