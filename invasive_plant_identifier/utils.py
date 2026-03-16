import os
import random
from typing import Tuple

import torch
from PIL import Image
from torchvision import datasets, transforms


from typing import Optional

def sync_label_manager_with_data(label_manager, data_dir: str) -> None:
    """Ensure the label manager contains every species folder found under
    ``data_dir``. Image files are used to detect valid class directories.

    This keeps the mapping used by the model in sync with the training
    dataset, avoiding `IndexError: Target ... out of bounds` when classes are
    added after the label manager was last updated.
    """
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    if item not in label_manager.labels:
                        label_manager.add_label(item)
                    break


def rebuild_label_manager_from_data(label_manager, data_dir: str) -> None:
    """Replace the label mapping with the exact class set found in ``data_dir``.

    The resulting class order matches ``torchvision.datasets.ImageFolder`` by
    sorting valid class folder names alphabetically.
    """
    classes = []
    for item in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, item)
        if not os.path.isdir(path):
            continue
        has_image = any(
            fname.lower().endswith((".jpg", ".jpeg", ".png")) for fname in os.listdir(path)
        )
        if has_image:
            classes.append(item)

    label_manager.labels = {name: idx for idx, name in enumerate(classes)}
    label_manager._save()


def remove_species_data(data_dir: str, species: str) -> None:
    """Delete the specified species folder and all contained images.

    Does not touch label or database metadata; caller should take care of
    updating those if desired. Raises ``FileNotFoundError`` if species is
    missing.
    """
    path = os.path.join(data_dir, species)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Species folder not found: {species}")
    import shutil

    shutil.rmtree(path)


def wipe_training_data(data_dir: str) -> None:
    """Remove all species subdirectories under ``data_dir``.

    The directory itself will be recreated if necessary.
    """
    import shutil

    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)


def create_imagefolder_datasets(
    root_dir: str,
    input_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
):
    """Create torchvision ImageFolder datasets for training and validation.

    Assumes that root_dir contains subdirectories for each class containing images.
    """
    transform_train = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    full_dataset = datasets.ImageFolder(root_dir, transform=transform_train)
    # split indices
    total = len(full_dataset)
    val_size = int(total * val_split)
    indices = list(range(total))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = transform_train
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = transform_val

    return train_dataset, val_dataset


def create_imagefolder_datasets_from_dirs(
    train_root: str,
    val_root: str,
    input_size: Tuple[int, int] = (224, 224),
):
    """Create torchvision datasets for a training and a separate validation directory.

    Both directories must contain the same class subdirectories.
    """
    transform_train = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_root, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_root, transform=transform_val)
    return train_dataset, val_dataset


def wipe_app_state(
    model_path: str, label_path: str, db_path: str, data_dir: str
) -> None:
    """Wipe all persisted application state.

    Removes the trained model weights, label mapping, database file, and all
    training image data.

    The caller is responsible for closing any open database connections first.
    """
    # delete files if they exist
    for path in (model_path, label_path, db_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    wipe_training_data(data_dir)
