from dataloader import data_loader
from model import Generator, Discriminator
from os.path import join
import argparse
import os
import torch
import helper
import matplotlib.pyplot as plt
import lera
import yaml


def output_fig(images_array, file_name):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name + ".jpg", bbox_inches="tight", pad_inches=0)
    plt.close()


def train(args):
    try:
        os.makedirs(args.save_img_path)
    except OSError:
        pass

    try:
        os.makedirs(args.weight_path)
    except OSError:
        pass

    lera.log_hyperparams(
        {
            "title": "hw2",
            "batch_size": args.bs,
            "epochs": args.epochs,
            "g_lr": args.g_lr,
            "d_lr": args.d_lr,
            "z_size": args.z_size,
        }
    )

    # dataset
    dataloader = data_loader(
        args.data_path, args.imgsize, args.bs, shuffle=True
    )

    # model
    generator = Generator(args.bs, args.imgsize, z_dim=args.z_size).cuda()
    discriminator = Discriminator(args.bs, args.imgsize).cuda()
    if args.pre_epochs != 0:
        generator.load_state_dict(
            torch.load(
                join(f"{args.weight_path}", f"generator_{args.pre_epochs}.pth")
            )
        )

        discriminator.load_state_dict(
            torch.load(
                join(
                    f"{args.weight_path}",
                    f"discriminator_{args.pre_epochs}.pth",
                )
            )
        )

    # optimizer
    g_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()), lr=args.g_lr
    )
    d_optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=args.d_lr,
    )

    # validate noise
    fixed_noise = torch.randn(9, args.z_size)
    fixed_noise = torch.tensor(fixed_noise).cuda()

    # train
    for epoch in range(args.pre_epochs, args.epochs):
        for i, data in enumerate(dataloader):
            discriminator.train()
            generator.train()
            # train discriminator
            if i % 5 == 0:
                d_optimizer.zero_grad()
                real_img = torch.tensor(data[0]).cuda() * 2 - 1  # (-1, 1)
                d__real, _, _ = discriminator(real_img)
                z = torch.randn(args.bs, args.z_size)
                z = torch.tensor(z).cuda()
                fake_img, _, _ = generator(z)
                d_fake, _, _ = discriminator(fake_img)

                # hinge loss
                d_loss_real = torch.nn.ReLU()(1.0 - d__real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()

                d_optimizer.step()
            # train generator
            g_optimizer.zero_grad()
            z = torch.randn(args.bs, args.z_size)
            z = torch.tensor(z).cuda()
            fake_img, _, _ = generator(z)
            g_fake, _, _ = discriminator(fake_img)

            # hinge loss
            g_loss = -g_fake.mean()
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                lera.log({"d_loss": d_loss.item(), "g_loss": g_loss.item()})
                print(
                    "[epoch:%4d/%4d %3d/%3d] \t d_loss: %0.6f \t g_loss: %0.6f"
                    % (
                        epoch + 1,
                        args.epochs,
                        i,
                        len(dataloader),
                        d_loss.item(),
                        g_loss.item(),
                    )
                )
                if i % 300 == 0:
                    validate(
                        generator, i, epoch, args.save_img_path, fixed_noise
                    )

        torch.save(
            discriminator.state_dict(),
            f"./{args.weight_path}/discriminator_{epoch+1}.pth",
        )
        torch.save(
            generator.state_dict(),
            f"./{args.weight_path}/generator_{epoch+1}.pth",
        )


def validate(generator, i, epoch, path, fixed_noise):
    generator.eval()
    with torch.no_grad():
        generated_images, _, _ = generator(fixed_noise)
        generated_images = generated_images.permute(0, 2, 3, 1)
        generated_images = generated_images.data.cpu().numpy()
        output_fig(generated_images, file_name=f"{path}/{epoch}_{i}_validate")
    generator.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["train"].items():
        setattr(args, key, value)
    train(args)
