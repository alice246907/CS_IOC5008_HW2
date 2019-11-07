from torchvision import transforms
import torchvision.datasets as dsets
import torch


def data_loader(root, image_size, batch_size, shuffle=True):

    img_transforms = transforms.Compose(
        [
            transforms.CenterCrop(160),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = dsets.ImageFolder(root, transform=img_transforms)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        drop_last=True,
    )
    return loader
