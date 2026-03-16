import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from invasive_plant_identifier.utils import create_imagefolder_datasets
from invasive_plant_identifier.labels import LabelManager
from invasive_plant_identifier.model import PlantClassifier


def create_dummy_dataset(root, classes=("a", "b"), count=5):
    for cls in classes:
        cls_dir = Path(root) / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"\x00")


def test_create_imagefolder_datasets(tmp_path):
    data_dir = tmp_path / "data"
    create_dummy_dataset(data_dir)
    train_ds, val_ds = create_imagefolder_datasets(str(data_dir), val_split=0.2)
    total = len(train_ds) + len(val_ds)
    assert total == 10
    # check that each dataset is a torch.utils.data.Dataset
    assert isinstance(train_ds, torch.utils.data.Dataset)
    assert isinstance(val_ds, torch.utils.data.Dataset)


def test_sync_label_manager_and_training(tmp_path):
    # simulate two species folders but only one label in manager
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for sp in ["a", "b"]:
        sp_dir = data_dir / sp
        sp_dir.mkdir()
        # create a real image file so ImageFolder picks it up
        img = Image.new("RGB", (10, 10), color="white")
        img.save(sp_dir / "img.jpg")

    # start with label manager containing only 'a'
    label_file = tmp_path / "labels.json"
    lm = LabelManager(str(label_file))
    lm.add_label("a")
    assert lm.num_classes() == 1

    # call the sync helper
    from invasive_plant_identifier.utils import sync_label_manager_with_data
    sync_label_manager_with_data(lm, str(data_dir))
    assert lm.num_classes() == 2
    assert set(lm.labels.keys()) == {"a", "b"}

    # try a quick training run – should not raise IndexError
    classifier = PlantClassifier(lm)
    train_ds, val_ds = create_imagefolder_datasets(str(data_dir), val_split=0.5)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2)
    # classifier.train should run without exceptions
    classifier.train(train_loader, val_loader, epochs=1)


def test_training_data_management(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    # create two species directories
    for sp in ["x", "y"]:
        d = base / sp
        d.mkdir()
        (d / "img.jpg").write_bytes(b"\x00")
    from invasive_plant_identifier.utils import remove_species_data, wipe_training_data
    # remove one species
    remove_species_data(str(base), "x")
    assert not (base / "x").exists()
    # wipe everything
    wipe_training_data(str(base))
    assert base.exists() and not any(base.iterdir())


def test_wipe_app_state(tmp_path):
    # create dummy model, labels, and db files
    model_path = tmp_path / "model.pth"
    label_path = tmp_path / "labels.json"
    db_path = tmp_path / "detections.db"
    (model_path).write_text("dummy")
    (label_path).write_text("{}")
    (db_path).write_text("dummy")

    # create some training images
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "classA").mkdir()
    (data_dir / "classA" / "img.jpg").write_bytes(b"\x00")

    from invasive_plant_identifier.utils import wipe_app_state
    wipe_app_state(str(model_path), str(label_path), str(db_path), str(data_dir))

    assert not model_path.exists()
    assert not label_path.exists()
    assert not db_path.exists()
    assert data_dir.exists() and not any(data_dir.iterdir())

