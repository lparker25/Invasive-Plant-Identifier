import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .labels import LabelManager


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlantClassifier:
    """Wrapper around a PyTorch model for plant species classification."""

    def __init__(self, label_manager: LabelManager, model_path: str = None):
        self.labels = label_manager
        self.device = get_device()
        self.model = self._build_model()
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        # use a pretrained ResNet18 by default for speed; can be swapped later
        num_classes = max(1, self.labels.num_classes())
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-4,
    ) -> Tuple[float, float]:
        """Fine-tune the model with provided dataloaders.

        Returns:
            Tuple containing (train_acc, val_acc) on the final epoch.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        best_val_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            running_corrects = 0
            running_total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                preds = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(preds == labels).item()
                running_total += inputs.size(0)
            train_acc = running_corrects / running_total if running_total else 0

            # validation
            self.model.eval()
            val_corrects = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    val_corrects += torch.sum(preds == labels).item()
                    val_total += inputs.size(0)
            val_acc = val_corrects / val_total if val_total else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # optionally save best model
            print(f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        return train_acc, val_acc

    def predict(
        self, image: Image.Image
    ) -> Tuple[str, float, float]:
        """Classify a PIL image.

        Returns a tuple of (species_name, confidence, time_taken).
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        start = time.time()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf_val = conf.item()
            idx_val = idx.item()
        elapsed = time.time() - start
        try:
            species = self.labels.get_name(idx_val)
        except KeyError:
            species = "other"
        return species, conf_val, elapsed

    def save(self, path: str):
        """Save the model weights and label mapping."""
        state = {
            "model_state": self.model.state_dict(),
            "labels": self.labels.labels,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load weights (and optionally labels) from disk.

        If the number of classes has changed, only load compatible weights
        from the feature trunk and reinitialize the final layer.
        """
        state = torch.load(path, map_location=self.device)

        # If the saved checkpoint carries label metadata, apply it first so
        # the model architecture is built to match the stored label set.
        if "labels" in state:
            self.labels.labels = state["labels"]
            self.labels._save()

        self.model = self._build_model()
        model_state = state.get("model_state", state)

        # Try to load; if sizes mismatch (e.g., due to new classes),
        # load only the compatible parts
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as e:
            if "size mismatch" in str(e).lower():
                # Filter out the final layer (fc.weight, fc.bias)
                compatible_state = {
                    k: v for k, v in model_state.items() if not k.startswith("fc.")
                }
                self.model.load_state_dict(compatible_state, strict=False)
                print(
                    "Loaded feature trunk only; final layer reinitialized due to class count change"
                )
            else:
                raise

        self.model.to(self.device)



# helper functions for convenience

def save_model(classifier: PlantClassifier, path: str):
    classifier.save(path)


def load_model(label_path: str, model_path: Optional[str] = None) -> PlantClassifier:
    lm = LabelManager(label_path)
    return PlantClassifier(lm, model_path=model_path)


def predict_image(path: str, classifier: PlantClassifier):
    img = Image.open(path).convert("RGB")
    return classifier.predict(img)
