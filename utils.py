import torch
import torchvision
from dataset import ColoredMangaDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("--> saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("--> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_dataset = ColoredMangaDataset(
        image_dir=train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = ColoredMangaDataset(
        image_dir=val_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            num_correct += torch.sum(torch.abs(preds - y) <= 0.1)
            num_pixels += torch.numel(preds)
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100}.2f")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    model.train()
