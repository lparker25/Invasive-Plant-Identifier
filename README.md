# Invasive Plant Identifier Prototype

This repository contains a Python-based prototype application for identifying
invasive plant species using local image recognition, logging detections in a
SQLite database, and visualizing data on a Streamlit web dashboard.

## Features

- **Image Recognition**: PyTorch model (ResNet18) fine-tuned for plant species
  classification. Supports images, folders, and webcam/video stream input.
- **Database**: SQLite database logs each detection with timestamp, confidence,
  species name, invasive flag, and optional GPS coordinates.
- **Dashboard**: Streamlit app provides a user-friendly interface with:
  - Heatmap of detections on Google Maps
  - Species counts (invasive vs non-invasive)
  - Test mode for image uploads and webcam
  - Training mode for adding new species and fine-tuning the model
  - Data editing and export capabilities
- **Modular Code**: Structured into modules (`model.py`, `db.py`, `labels.py`,
  `utils.py`) with unit tests using `pytest`.

## Setup

1. **Create a Python environment** (recommended using `venv` or `conda`):

   ```bash
   python -m venv venv
   source venv/bin/activate    # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run tests** to ensure everything is working:

   ```bash
   pytest
   ```

3. **Start the dashboard**:

   ```bash
   streamlit run app.py
   ```

4. **Model files and labels** will be stored in the project root by default:
   - `labels.json`: contains species-to-index mapping.
   - `model.pth`: serialized PyTorch model.
   - `detections.db`: SQLite database file.

## Usage

Refer to the dashboard instructions when the Streamlit app opens. The sidebar
lets you switch between `Identification`, `Training`, and `Database` modes.

### Identification Mode

Upload images or use your webcam to identify plant species. Detections are
logged in the database. Confidence below 95\% will record the species as
"unknown".

### Training Mode

Add new species or upload additional photos for existing ones. Specify a label
and invasive status, then fine-tune the model for a few epochs. Training

The training pane also contains a **Manage training data** section where you
can delete individual species folders or wipe the entire dataset if you want
to start over. These operations immediately remove files under `data/` and
affect subsequent training runs.

Each detection record now stores the **image filename** that produced it, and
the database table includes an **Is Correct** checkbox column. You can edit
records to mark misclassifications for later review or filtering. This works
via the manual edit control in the Database tab.

The app can ask for your browser's geolocation when you click the **Use my
location** button; latitude/longitude values are logged with each detection.
If your Streamlit version does not support query parameters, the location
feature gracefully degrades (no error will be thrown).
data is stored in `data/<species>/`.

### Database

Inspect, edit, or delete entries. Export detections to CSV. Mark species as
invasive or non-invasive.

## Extensibility

The code is designed for later integration with drone GPS data and models can
be exported via `torchscript` or ONNX for performance optimization.

## Notes

- This prototype does not rely on any external APIs for model inference
  (except Google Maps for dashboard heatmaps).
- Location data is optional and may be set to `"N/A"` during testing.

## License

MIT
