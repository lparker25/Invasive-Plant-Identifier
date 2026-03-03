import os
from PIL import Image
import torch

import pytest

from invasive_plant_identifier.model import PlantClassifier
from invasive_plant_identifier.labels import LabelManager


def test_model_prediction(tmp_path):
    # create temporary label manager with one label
    label_file = tmp_path / "labels.json"
    lm = LabelManager(str(label_file))
    lm.add_label("dummy")
    classifier = PlantClassifier(lm)
    # create a simple white image
    img = Image.new("RGB", (224, 224), color="white")
    species, conf, elapsed = classifier.predict(img)
    assert species in ["dummy", "unknown"]
    assert 0.0 <= conf <= 1.0
    assert elapsed >= 0

    # saving and loading works
    model_path = tmp_path / "model.pth"
    classifier.save(str(model_path))
    assert model_path.exists()
    loaded = PlantClassifier(lm, model_path=str(model_path))
    species2, conf2, _ = loaded.predict(img)
    assert 0.0 <= conf2 <= 1.0

