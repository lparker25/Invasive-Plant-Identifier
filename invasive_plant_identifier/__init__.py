"""Package for invasive plant identification prototype."""

from .model import PlantClassifier, load_model, save_model, predict_image
from .db import Database
from .labels import LabelManager
