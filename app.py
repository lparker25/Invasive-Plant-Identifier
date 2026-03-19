import io
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, List, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from invasive_plant_identifier.db import Database
from invasive_plant_identifier.labels import LabelManager
from invasive_plant_identifier.model import PlantClassifier, predict_image, load_model
from invasive_plant_identifier.utils import create_imagefolder_datasets

# --- constants and paths --------------------------------------------------
# File locations for persistent storage. These live in the project root so that
# the Streamlit app, tests, and modules all refer to the same state. The
# database is a lightweight SQLite file; labels.json holds the species mapping;
# model.pth stores the serialized PyTorch weights.
MODEL_PATH = "model.pth"
LABEL_PATH = "labels.json"
DB_PATH = "detections.db"
DATA_DIR = "data"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# persistent objects via session state
if "label_manager" not in st.session_state:
    st.session_state.label_manager = LabelManager(LABEL_PATH)
if "classifier" not in st.session_state:
    st.session_state.classifier = PlantClassifier(st.session_state.label_manager, model_path=MODEL_PATH)
if "database" not in st.session_state:
    st.session_state.database = Database(DB_PATH)
if "pending_species_flags" not in st.session_state:
    st.session_state.pending_species_flags = {}
if "selected_identification_run_id" not in st.session_state:
    st.session_state.selected_identification_run_id = st.session_state.database.get_latest_run_id()
if "selected_db_run_id" not in st.session_state:
    st.session_state.selected_db_run_id = st.session_state.database.get_latest_run_id()


# utility helpers ----------------------------------------------------------


def _remove_file(path: str) -> None:
    """Delete a file if it exists. Ignore any errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _rerun_app() -> None:
    """Rerun app across Streamlit versions that use different APIs."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def reset_app_state() -> None:
    """Wipe all persisted app state and restart the Streamlit app."""
    # close database connection before removing the file (especially on Windows)
    db = st.session_state.get("database")
    if db is not None:
        try:
            db.close()
        except Exception:
            pass

    # delete stored artifacts and training data
    from invasive_plant_identifier.utils import wipe_app_state

    wipe_app_state(MODEL_PATH, LABEL_PATH, DB_PATH, DATA_DIR)

    # clear in-memory state and rerun
    st.session_state.clear()
    _rerun_app()


def classify_and_log(
    image: Image.Image,
    gps: Tuple[Any, Any] = ("N/A", "N/A"),
    threshold: float = 0.95,
    image_id: str = "N/A",
    run_id: int | None = None,
) -> Tuple[str, float, float]:
    """Classify a PIL image, log the result in the database.

    The function applies the current classifier to the supplied image. If the
    returned confidence is below the threshold it forces the species to
    "other" and sets confidence to zero. The detection is inserted into the
    SQLite database along with timestamp, analysis time, optional GPS, and
    image filename.

    Returns a tuple of (species, confidence, elapsed_seconds) for display in
    the UI.
    """
    species, confidence, elapsed = st.session_state.classifier.predict(image)
    if confidence < threshold:
        # Low confidence means we treat it as "other" (i.e., not a recognized species)
        species = "other"
        confidence = 0.0
    # determine invasive flag
    invasive_flag = False
    try:
        # lookup from species table
        c = st.session_state.database.conn.cursor()
        c.execute("SELECT is_invasive FROM species WHERE name = ?", (species,))
        r = c.fetchone()
        if r and r[0] == 1:
            invasive_flag = True
    except Exception:
        invasive_flag = False
    st.session_state.database.log_detection(
        datetime=datetime.utcnow().isoformat(),
        analysis_time=elapsed,
        confidence_score=confidence,
        species=species,
        is_invasive=invasive_flag,
        latitude=gps[0],
        longitude=gps[1],
        image_id=image_id,
        run_id=run_id,
    )
    return species, confidence, elapsed


def _snapshot_uploaded_files(uploaded_files) -> List[Tuple[str, bytes]]:
    """Copy uploaded files into immutable in-memory blobs for repeatable runs."""
    snapshots: List[Tuple[str, bytes]] = []
    for uf in uploaded_files or []:
        snapshots.append((uf.name, uf.getvalue()))
    return snapshots


def _process_uploaded_snapshot(
    snapshots: List[Tuple[str, bytes]],
    threshold: float,
    gps: Tuple[Any, Any],
    run_id: int,
) -> Tuple[int, int]:
    """Process one complete pass of uploaded files for a specific run id."""
    import zipfile

    processed_count = 0
    low_conf_count = 0

    for filename, blob in snapshots:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for info in z.infolist():
                    if info.is_dir() or not _is_image_file(info.filename):
                        continue
                    with z.open(info) as imgf:
                        image = Image.open(imgf).convert("RGB")
                        species, conf, elapsed = classify_and_log(
                            image,
                            threshold=threshold,
                            image_id=info.filename,
                            gps=gps,
                            run_id=run_id,
                        )
                        st.image(image, caption=f"{species} ({conf:.2f})", use_column_width=True)
                        if species == "other":
                            low_conf_count += 1
                            st.warning(
                                f"⚠️ {info.filename}: **OTHER (LOW CONFIDENCE)** - Confidence below threshold"
                            )
                        else:
                            st.write(f"{info.filename}: {species} ({conf:.2f}) analyzed in {elapsed:.2f}s")
                        processed_count += 1
        elif _is_image_file(filename):
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            species, conf, elapsed = classify_and_log(
                image,
                threshold=threshold,
                image_id=filename,
                gps=gps,
                run_id=run_id,
            )
            st.image(image, caption=f"{species} ({conf:.2f})", use_column_width=True)
            if species == "other":
                low_conf_count += 1
                st.warning("⚠️ **OTHER (LOW CONFIDENCE)** - Confidence below threshold")
            else:
                st.write(f"Analyzed in {elapsed:.2f}s")
            processed_count += 1

    return processed_count, low_conf_count


def _render_run_summary_panel(run_id: int, run_label: str) -> None:
    """Render a compact summary for a single identification run."""
    rows = st.session_state.database.get_all_detections(run_id=run_id)
    if not rows:
        st.caption(f"{run_label}: no detections yet.")
        return

    df_run = pd.DataFrame([dict(r) for r in rows])
    total = len(df_run)
    known = int((df_run["species"] != "other").sum()) if "species" in df_run.columns else 0
    other = int((df_run["species"] == "other").sum()) if "species" in df_run.columns else 0

    with st.expander(f"Run summary: {run_label}", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Known", known)
        c3.metric("Other", other)

        species_counts = st.session_state.database.get_species_counts(run_id=run_id)
        if species_counts:
            summary_df = pd.DataFrame(
                species_counts,
                columns=["Species", "Detections", "Invasive Detections"],
            ).sort_values("Detections", ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _count_images_in_dir(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if _is_image_file(f)])


def _save_uploaded_files_for_species(uploaded_files, species: str, split: str) -> int:
    """Save uploaded image/zip content to data/<species>/<split> and return count."""
    if not uploaded_files:
        return 0

    split_dir = os.path.join(DATA_DIR, species, split)
    os.makedirs(split_dir, exist_ok=True)
    saved_count = 0

    for uf in uploaded_files:
        if uf.name.lower().endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(uf) as z:
                for info in z.infolist():
                    if info.is_dir() or not _is_image_file(info.filename):
                        continue
                    timestamp = time.time_ns()
                    filename = os.path.basename(info.filename) or f"image_{saved_count}.jpg"
                    target = os.path.join(split_dir, f"{timestamp}_{filename}")
                    with z.open(info) as imgf, open(target, "wb") as out:
                        out.write(imgf.read())
                    saved_count += 1
        elif _is_image_file(uf.name):
            timestamp = time.time_ns()
            target = os.path.join(split_dir, f"{timestamp}_{uf.name}")
            with open(target, "wb") as out:
                out.write(uf.read())
            saved_count += 1

    return saved_count


def _copy_species_training_images(source_species_dir: str, tmp_root: str, species: str) -> int:
    """Copy train images (or legacy flat images) into a temporary ImageFolder class dir."""
    import shutil

    train_dir = os.path.join(source_species_dir, "train")
    src_dir = train_dir if os.path.isdir(train_dir) else source_species_dir
    if not os.path.isdir(src_dir):
        return 0

    dst_dir = os.path.join(tmp_root, species)
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(src_dir):
        if not _is_image_file(fname):
            continue
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
        copied += 1
    if copied == 0 and os.path.isdir(dst_dir):
        os.rmdir(dst_dir)
    return copied


def _copy_species_validation_images(source_species_dir: str, tmp_root: str, species: str) -> int:
    """Copy val images from data/<species>/val into a temporary ImageFolder class dir."""
    import shutil

    src_dir = os.path.join(source_species_dir, "val")
    if not os.path.isdir(src_dir):
        return 0

    dst_dir = os.path.join(tmp_root, species)
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(src_dir):
        if not _is_image_file(fname):
            continue
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
        copied += 1
    if copied == 0 and os.path.isdir(dst_dir):
        os.rmdir(dst_dir)
    return copied


def show_heatmap(df: pd.DataFrame):
    # check for required columns
    required_cols = ["latitude", "longitude", "species", "is_invasive", "confidence_score"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning(f"Cannot display heatmap: missing columns {missing}")
        return
    
    # filter entries with valid floats
    df_valid = df[(df["latitude"] != "N/A") & (df["longitude"] != "N/A")]
    if df_valid.empty:
        st.info("No geolocated entries to display.")
        return
    # convert types
    df_valid = df_valid.copy()
    df_valid["latitude"] = pd.to_numeric(df_valid["latitude"], errors="coerce")
    df_valid["longitude"] = pd.to_numeric(df_valid["longitude"], errors="coerce")
    df_valid.dropna(subset=["latitude", "longitude"], inplace=True)

    # create folium map centered at mean location
    m = folium.Map(location=[df_valid["latitude"].mean(), df_valid["longitude"].mean()], zoom_start=6)
    for _, row in df_valid.iterrows():
        color = "red" if row["is_invasive"] == 1 else "green"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{row['species']} ({row['confidence_score']:.2f})",
        ).add_to(m)
    st.components.v1.html(m._repr_html_(), height=500)


# --- streamlit layout -----------------------------------------------------

st.title("Invasive Plant Identifier")
# handle optional geolocation query parameters
try:
    params = st.experimental_get_query_params()
except AttributeError:
    params = {}
if "lat" in params and "lng" in params:
    try:
        lat = float(params["lat"][0])
        lng = float(params["lng"][0])
        st.session_state.gps = (lat, lng)
    except ValueError:
        st.session_state.gps = ("N/A", "N/A")
# sidebar navigation allows switching between the main modes of the app
mode = st.sidebar.radio("Mode", ["Identification", "Training", "Database"])

# app reset / reload
st.sidebar.markdown("---")
if st.sidebar.button("Reload app (wipe all stored data)"):
    st.session_state["reload_pending"] = True

if st.session_state.get("reload_pending"):
    st.sidebar.warning(
        "⚠️ This will permanently delete training images, model weights, labels, and all database records."
    )
    if st.sidebar.button("Confirm reload", key="confirm_reload"):
        reset_app_state()
    if st.sidebar.button("Cancel", key="cancel_reload"):
        st.session_state["reload_pending"] = False

if mode == "Identification":
    st.header("Identification / Test")
    st.write("Upload an image or use webcam to test the model.")

    # location permission button
    if "gps" not in st.session_state:
        loc_html = '''
        <button id="getloc">Use my location</button>
        <script>
        document.getElementById('getloc').onclick = function(){
          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(pos){
              const lat = pos.coords.latitude;
              const lng = pos.coords.longitude;
              window.location.href = window.location.pathname + '?lat=' + lat + '&lng=' + lng;
            }, function(err){alert('Location error: '+err.message);});
          } else {alert('Geolocation not supported');}
        }
        </script>
        '''
        st.components.v1.html(loc_html, height=80)
    else:
        st.write(f"Current GPS: {st.session_state.gps[0]}, {st.session_state.gps[1]}")

    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence threshold for species identification (lower = more 'other' detections)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="If model confidence is below this threshold, the plant will be marked as 'other'"
    )

    runs = st.session_state.database.list_runs()
    run_ids = [int(r["run_id"]) for r in runs]
    if st.session_state.selected_identification_run_id not in run_ids:
        st.session_state.selected_identification_run_id = st.session_state.database.get_latest_run_id()

    run_labels = {
        int(r["run_id"]): f"{r['run_label']} (id {r['run_id']}, {r['source_type']})"
        for r in runs
    }
    selected_ident_run_id = st.selectbox(
        "Single-capture target run",
        options=run_ids,
        format_func=lambda rid: run_labels.get(rid, f"Run id {rid}"),
        index=run_ids.index(st.session_state.selected_identification_run_id),
    )
    st.session_state.selected_identification_run_id = int(selected_ident_run_id)
    _render_run_summary_panel(
        st.session_state.selected_identification_run_id,
        run_labels.get(
            st.session_state.selected_identification_run_id,
            f"Run {st.session_state.selected_identification_run_id}",
        ),
    )

    run_count = st.number_input(
        "Uploaded-image passes",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Each pass creates a new run (Run 1, Run 2, ...) and logs detections separately.",
    )

    uploaded_files = st.file_uploader(
        "Choose image(s) or a zip file", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Run uploaded identification passes"):
            snapshots = _snapshot_uploaded_files(uploaded_files)
            if not snapshots:
                st.warning("No uploaded files available for processing.")
            else:
                total_processed = 0
                for _ in range(int(run_count)):
                    run_id, run_label = st.session_state.database.create_run(source_type="uploaded")
                    st.session_state.selected_identification_run_id = run_id
                    st.session_state.selected_db_run_id = run_id
                    st.subheader(f"{run_label}")
                    processed_count, low_conf_count = _process_uploaded_snapshot(
                        snapshots,
                        threshold=confidence_threshold,
                        gps=st.session_state.get("gps", ("N/A", "N/A")),
                        run_id=run_id,
                    )
                    total_processed += processed_count
                    st.info(
                        f"{run_label}: processed {processed_count} image(s), low-confidence detections: {low_conf_count}."
                    )
                    _render_run_summary_panel(run_id, run_label)
                st.success(
                    f"Completed {int(run_count)} run(s). Total detections logged: {total_processed}."
                )

    st.write("---")
    st.write("**Webcam / video input**")
    if st.button("Capture from webcam"):
        try:
            import cv2

            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                species, conf, elapsed = classify_and_log(
                    img,
                    threshold=confidence_threshold,
                    image_id="webcam_capture",
                    gps=st.session_state.get("gps", ("N/A", "N/A")),
                    run_id=st.session_state.selected_identification_run_id,
                )
                st.image(frame, channels="BGR", caption=f"{species} ({conf:.2f})")
                if species == "other":
                    st.warning(f"⚠️ **OTHER (LOW CONFIDENCE)** - Confidence below threshold")
                else:
                    st.write(f"Analyzed in {elapsed:.2f}s")
            else:
                st.error("Failed to capture image from webcam.")
        except Exception as e:
            st.error(f"Webcam error: {e}")


elif mode == "Training":
    st.header("Training / Add Species")
    st.write("Build a species dataset one species at a time, then train on selected species.")
    
    st.subheader("Step 1: Save one species dataset")
    st.write("Provide a species name, upload training images, optionally upload validation images, then click the add button.")
    
    uploaded = st.file_uploader(
        "Training images (image files or zip)",
        type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True,
        key="train_upload",
    )
    species_name = st.text_input(
        "Species name", key="train_species_name"
    )
    invasive_flag = st.checkbox("Mark as invasive", value=False, key="train_invasive")

    st.write("---")
    st.subheader("Optional: Validation images")
    st.write("These are saved with the species and used as a separate validation set during training.")
    val_uploaded = st.file_uploader(
        "Validation images (optional)", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True, key="val_upload"
    )
    
    if st.button("Add species and clear form"):
        if not uploaded:
            st.warning("Please provide training images.")
        elif not species_name.strip():
            st.warning("Please provide a species name.")
        else:
            clean_species = species_name.strip()
            train_saved = _save_uploaded_files_for_species(uploaded, clean_species, "train")
            val_saved = _save_uploaded_files_for_species(val_uploaded, clean_species, "val")

            if train_saved == 0:
                st.warning("No valid image files were found in the training upload.")
            else:
                # Keep invasive flags in session until training starts.
                st.session_state.pending_species_flags[clean_species] = bool(invasive_flag)
                st.success(
                    f"Saved species '{clean_species}' with {train_saved} training images and {val_saved} validation images."
                )

                # Clear form inputs so users can immediately stage the next species.
                try:
                    st.session_state.update(
                        {
                            "train_upload": None,
                            "train_species_name": "",
                            "train_invasive": False,
                            "val_upload": None,
                        }
                    )
                except Exception:
                    pass

                _rerun_app()
    
    st.write("---")
    st.subheader("Step 2: Select species to train on")
    
    # Get all available species folders
    available_species = []
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if not os.path.isdir(item_path):
                continue

            train_count = _count_images_in_dir(os.path.join(item_path, "train"))
            legacy_count = _count_images_in_dir(item_path)
            val_count = _count_images_in_dir(os.path.join(item_path, "val"))
            total_train = train_count if train_count > 0 else legacy_count

            if total_train > 0:
                available_species.append((item, total_train, val_count))
    
    if not available_species:
        st.info("No species with training images found. Upload some images first.")
    else:
        st.write(f"Found {len(available_species)} species with training images:")
        for sp, train_count, val_count in available_species:
            st.write(f"  - {sp}: {train_count} train images, {val_count} val images")
        
        selected_species = st.multiselect(
            "Select species to train on (all selected species will be trained together)",
            [sp[0] for sp in available_species],
            default=[sp[0] for sp in available_species]
        )
        
        if selected_species:
            epochs = st.slider("Training epochs", min_value=1, max_value=20, value=5)
            if st.button("Start training on selected species"):
                st.write(f"Training model on: {', '.join(selected_species)}")

                # build temporary directory containing only selected species
                import shutil, tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    copied_train_counts = {}
                    for sp in selected_species:
                        src = os.path.join(DATA_DIR, sp)
                        copied_train_counts[sp] = _copy_species_training_images(src, tmpdir, sp)

                    no_train_species = [sp for sp, count in copied_train_counts.items() if count == 0]
                    if no_train_species:
                        st.error(
                            "No training images found for: " + ", ".join(no_train_species)
                        )
                        st.stop()

                    # rebuild labels from the exact temporary training dataset so
                    # class indices match ImageFolder ordering with no stale labels
                    from invasive_plant_identifier.utils import rebuild_label_manager_from_data

                    rebuild_label_manager_from_data(st.session_state.label_manager, tmpdir)

                    # rebuild classifier after syncing labels so it matches available classes
                    st.session_state.classifier = PlantClassifier(
                        st.session_state.label_manager,
                        model_path=MODEL_PATH,
                        load_checkpoint_labels=False,
                    )

                    # Register species in DB only when they are used for training.
                    for sp in selected_species:
                        is_invasive = bool(st.session_state.pending_species_flags.get(sp, False))
                        st.session_state.database.add_species(sp, is_invasive)
                        st.session_state.database.set_invasive(sp, is_invasive)

                    species_with_val = [
                        sp for sp in selected_species if _count_images_in_dir(os.path.join(DATA_DIR, sp, "val")) > 0
                    ]

                    if species_with_val and len(species_with_val) == len(selected_species):
                        from invasive_plant_identifier.utils import create_imagefolder_datasets_from_dirs

                        with tempfile.TemporaryDirectory() as val_dir:
                            for sp in selected_species:
                                src = os.path.join(DATA_DIR, sp)
                                _copy_species_validation_images(src, val_dir, sp)

                            # rebuild classifier after label sync so it matches available labels
                            st.session_state.classifier = PlantClassifier(
                                st.session_state.label_manager,
                                model_path=MODEL_PATH,
                                load_checkpoint_labels=False,
                            )

                            train_ds, val_ds = create_imagefolder_datasets_from_dirs(tmpdir, val_dir)
                            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
                            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)
                            st.write(
                                f"Training on {len(train_ds)} images, validating on {len(val_ds)} images."
                            )
                            with st.spinner("Training model..."):
                                train_acc, val_acc = st.session_state.classifier.train(
                                    train_loader, val_loader, epochs=epochs
                                )
                    else:
                        if species_with_val:
                            st.warning(
                                "Some selected species are missing validation images; using automatic train/validation split instead."
                            )
                        train_ds, val_ds = create_imagefolder_datasets(tmpdir)
                        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
                        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)
                        st.write(
                            f"Training on {len(train_ds)} images, validating on {len(val_ds)} images."
                        )
                        with st.spinner("Training model..."):
                            train_acc, val_acc = st.session_state.classifier.train(
                                train_loader, val_loader, epochs=epochs
                            )

                st.success(f"Finished training: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                st.session_state.classifier.save(MODEL_PATH)

        # management of existing training data
        st.write("---")
        st.subheader("Manage training data")
        if os.path.exists(DATA_DIR):
            current_species = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        else:
            current_species = []
        if current_species:
            to_delete = st.multiselect("Species to remove", current_species)
            if st.button("Delete selected species") and to_delete:
                from invasive_plant_identifier.utils import remove_species_data
                for sp in to_delete:
                    try:
                        remove_species_data(DATA_DIR, sp)
                        st.session_state.pending_species_flags.pop(sp, None)
                        st.success(f"Deleted {sp}")
                    except Exception as e:
                        st.error(f"Failed to delete {sp}: {e}")
                _rerun_app()
        if st.button("Wipe all training data"):
            confirm = st.checkbox("I understand this will delete every training image", key="wipe_confirm")
            if confirm and st.button("Confirm wipe", key="wipe_confirm_btn"):
                from invasive_plant_identifier.utils import wipe_training_data
                wipe_training_data(DATA_DIR)
                st.session_state.pending_species_flags = {}
                st.success("All training data removed")
                _rerun_app()

elif mode == "Database":
    st.header("Database")
    runs = st.session_state.database.list_runs()
    run_options = ["all"] + [int(r["run_id"]) for r in runs]

    if st.session_state.selected_db_run_id not in run_options:
        st.session_state.selected_db_run_id = st.session_state.database.get_latest_run_id()

    selected_db_scope = st.selectbox(
        "View run",
        options=run_options,
        format_func=lambda x: (
            "All runs"
            if x == "all"
            else next(
                (
                    f"{r['run_label']} (id {r['run_id']}, {r['source_type']})"
                    for r in runs
                    if int(r["run_id"]) == int(x)
                ),
                f"Run id {x}",
            )
        ),
        index=run_options.index(st.session_state.selected_db_run_id),
    )

    selected_db_run_id = None if selected_db_scope == "all" else int(selected_db_scope)
    if selected_db_run_id is not None:
        st.session_state.selected_db_run_id = selected_db_run_id

    rows = st.session_state.database.get_all_detections(run_id=selected_db_run_id)
    if not rows:
        st.info("No detections yet.")
        df = pd.DataFrame()  # empty dataframe
    else:
        # Convert sqlite3.Row objects to dict for proper column naming
        data = [dict(row) for row in rows]
        df = pd.DataFrame(data)
        # Ensure proper column order and naming
        column_order = ["id", "run_id", "datetime", "analysis_time", "confidence_score", "species", "is_invasive", "image_id", "is_correct", "latitude", "longitude"]
        # only use columns that exist in the df
        df = df[[col for col in column_order if col in df.columns]]
    
    if df.empty:
        st.info("No detections yet.")
    else:
        # create a display version with better column names
        df_display = df.copy()
        df_display.columns = [col.replace("_", " ").title() for col in df_display.columns]
        # convert is_correct column to boolean if present
        if "Is Correct" in df_display.columns:
            df_display["Is Correct"] = df_display["Is Correct"].astype(bool)
        
        # show editable table so users can correct mistakes
        st.subheader("Detection Records")
        edited = st.data_editor(df_display, use_container_width=True)
        if st.button("Apply table changes"):
            # Map display names back to original column names
            display_to_original = {col.replace("_", " ").title(): col for col in df.columns}
            # persist any differences back to the database
            for idx, row in edited.iterrows():
                orig = df.iloc[idx]
                changes = {}
                for display_col, value in row.items():
                    original_col = display_to_original.get(display_col, display_col)
                    if original_col in df.columns and value != orig[original_col]:
                        changes[original_col] = value
                if changes:
                    # convert booleans if necessary
                    if "is_invasive" in changes:
                        changes["is_invasive"] = int(changes["is_invasive"])
                    st.session_state.database.update_detection(
                        int(orig["id"]), run_id=selected_db_run_id, **changes
                    )
            st.success("Database updated")
        # deletion UI
        if "id" in df.columns:
            ids_to_delete = st.multiselect(
                "Select rows to delete (by id)", df["id"].tolist()
            )
            if st.button("Delete selected rows") and ids_to_delete:
                for rid in ids_to_delete:
                    st.session_state.database.delete_detection(int(rid), run_id=selected_db_run_id)
                st.success("Rows deleted")
                _rerun_app()
        if st.button("Export CSV"):
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            st.session_state.database.export_csv(tmpfile.name, run_id=selected_db_run_id)
            run_suffix = "all" if selected_db_run_id is None else f"run_{selected_db_run_id}"
            st.download_button(
                "Download data",
                open(tmpfile.name, "rb"),
                file_name=f"detections_{run_suffix}.csv",
            )
    st.write("---")
    st.subheader("Species management")
    species_list = [r[0] for r in st.session_state.database.conn.execute("SELECT name FROM species").fetchall()]
    selected = st.selectbox("Select species to mark invasive/non-invasive", [""] + species_list)
    if selected:
        current = st.session_state.database.conn.execute(
            "SELECT is_invasive FROM species WHERE name = ?", (selected,)
        ).fetchone()[0]
        is_other = selected == "other"
        if is_other:
            st.info("'other' species cannot be marked as invasive. It represents low-confidence detections.")
        else:
            new_flag = st.checkbox("Invasive", value=bool(current))
            if st.button("Update invasive status"):
                st.session_state.database.set_invasive(selected, new_flag)
                st.success("Updated")

    st.write("---")
    st.subheader("Visualizations")
    if not df.empty:
        st.write("**Detection Statistics**")
        # guard against missing columns
        if "species" in df.columns:
            other_count = len(df[df["species"] == "other"])
            known_count = len(df[df["species"] != "other"])
            st.metric("Known Species Detections", known_count)
            st.metric("Other/Unknown Detections", other_count)
        
        show_heatmap(df)
        
        # only chart if both required columns exist
        if "species" in df.columns and "is_invasive" in df.columns:
            counts = df.groupby(["species", "is_invasive"]).size().unstack(fill_value=0)
            st.bar_chart(counts)

        # option to wipe database
        st.write("---")
        st.subheader("Database maintenance")
        col1, col2 = st.columns(2)
        with col1:
            if selected_db_run_id is not None and st.button("🧹 Clear selected run", use_container_width=True):
                st.session_state["db_run_wipe_pending"] = True
            if st.button("🗑️ Clear all detections", use_container_width=True):
                st.session_state["db_wipe_pending"] = True

        if st.session_state.get("db_run_wipe_pending", False) and selected_db_run_id is not None:
            st.warning("⚠️ This will delete all detections from the selected run only.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, clear selected run", use_container_width=True, key="confirm_run_wipe"):
                    st.session_state.database.clear_detections(run_id=selected_db_run_id)
                    st.session_state["db_run_wipe_pending"] = False
                    st.success("Selected run detections removed")
                    _rerun_app()
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_run_wipe"):
                    st.session_state["db_run_wipe_pending"] = False
                    _rerun_app()
        
        # show confirmation only if wipe was requested
        if st.session_state.get("db_wipe_pending", False):
            st.warning("⚠️ This will permanently delete **all** detection records from the database. This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, delete all", use_container_width=True, key="confirm_wipe"):
                    st.session_state.database.clear_detections()
                    st.session_state["db_wipe_pending"] = False
                    st.session_state["db_run_wipe_pending"] = False
                    st.success("All detections removed from database")
                    _rerun_app()
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_wipe"):
                    st.session_state["db_wipe_pending"] = False
                    _rerun_app()
