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

# ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# persistent objects via session state
if "label_manager" not in st.session_state:
    st.session_state.label_manager = LabelManager(LABEL_PATH)
if "classifier" not in st.session_state:
    st.session_state.classifier = PlantClassifier(st.session_state.label_manager, model_path=MODEL_PATH)
if "database" not in st.session_state:
    st.session_state.database = Database(DB_PATH)


# utility helpers ----------------------------------------------------------


def _remove_file(path: str) -> None:
    """Delete a file if it exists. Ignore any errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


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
    st.experimental_rerun()


def classify_and_log(
    image: Image.Image,
    gps: Tuple[Any, Any] = ("N/A", "N/A"),
    threshold: float = 0.95,
    image_id: str = "N/A",
) -> Tuple[str, float, float]:
    """Classify a PIL image, log the result in the database.

    The function applies the current classifier to the supplied image. If the
    returned confidence is below the threshold it forces the species to
    "unknown" and sets confidence to zero. The detection is inserted into the
    SQLite database along with timestamp, analysis time, optional GPS, and
    image filename.

    Returns a tuple of (species, confidence, elapsed_seconds) for display in
    the UI.
    """
    species, confidence, elapsed = st.session_state.classifier.predict(image)
    if confidence < threshold:
        species = "unknown"
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
    )
    return species, confidence, elapsed


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
        "Confidence threshold for species identification (lower = more 'unknown' detections)",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="If model confidence is below this threshold, the plant will be marked as 'unknown'"
    )

    uploaded_files = st.file_uploader(
        "Choose image(s) or a zip file", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name.lower().endswith(".zip"):
                # process zip archive as folder input
                import zipfile

                with zipfile.ZipFile(uf) as z:
                    for info in z.infolist():
                        if info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                            with z.open(info) as imgf:
                                image = Image.open(imgf).convert("RGB")
                                species, conf, elapsed = classify_and_log(
                                    image,
                                    threshold=confidence_threshold,
                                    image_id=info.filename,
                                    gps=st.session_state.get("gps", ("N/A", "N/A")),
                                )
                                st.image(image, caption=f"{species} ({conf:.2f})", use_column_width=True)
                                if species == "unknown":
                                    st.warning(f"⚠️ {info.filename}: **UNKNOWN SPECIES** - Confidence below threshold")
                                else:
                                    st.write(f"{info.filename}: {species} ({conf:.2f}) analyzed in {elapsed:.2f}s")
            else:
                image = Image.open(uf).convert("RGB")
                species, conf, elapsed = classify_and_log(
                    image,
                    threshold=confidence_threshold,
                    image_id=uf.name,
                    gps=st.session_state.get("gps", ("N/A", "N/A")),
                )
                st.image(image, caption=f"{species} ({conf:.2f})", use_column_width=True)
                if species == "unknown":
                    st.warning(f"⚠️ **UNKNOWN SPECIES** - Confidence below threshold")
                else:
                    st.write(f"Analyzed in {elapsed:.2f}s")

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
                )
                st.image(frame, channels="BGR", caption=f"{species} ({conf:.2f})")
                if species == "unknown":
                    st.warning(f"⚠️ **UNKNOWN SPECIES** - Confidence below threshold")
                else:
                    st.write(f"Analyzed in {elapsed:.2f}s")
            else:
                st.error("Failed to capture image from webcam.")
        except Exception as e:
            st.error(f"Webcam error: {e}")


elif mode == "Training":
    st.header("Training / Add Species")
    st.write("Upload images for one or more species and fine-tune the model.")
    
    st.subheader("Step 1: Add or organize training images")
    st.write("You can either:")
    st.write("- Upload images for a new species")
    st.write("- Upload a zip file with folders organized by species (each folder is a species)")
    
    uploaded = st.file_uploader(
        "Select training images (can be zip archive)", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True
    )
    species_name = st.text_input("Species name (leave blank if uploading zip with folders)")
    invasive_flag = st.checkbox("Mark as invasive", value=False)

    st.write("---")
    st.subheader("Optional: Validation images")
    st.write("Upload additional images that will be used as a separate validation set (will not be used for training).")
    val_uploaded = st.file_uploader(
        "Validation images (optional) - can be zip or image files", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True, key="val_upload"
    )
    val_species_name = st.text_input(
        "Validation species name (leave blank if uploading zip with folders)", key="val_species_name"
    )
    
    if st.button("Upload images"):
        if not uploaded:
            st.warning("Please provide images.")
        elif not species_name and not any(uf.name.lower().endswith(".zip") for uf in uploaded):
            st.warning("Please provide a species name or upload a zip file with organized folders.")
        else:
            for uf in uploaded:
                if uf.name.lower().endswith(".zip"):
                    # zip file may contain multiple species folders
                    import zipfile
                    with zipfile.ZipFile(uf) as z:
                        for info in z.infolist():
                            if info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                                # Extract folder/species structure
                                parts = info.filename.split("/")
                                if len(parts) >= 2:
                                    spec_name = parts[0]
                                else:
                                    spec_name = species_name if species_name else "uploaded"
                                
                                species_dir = os.path.join(DATA_DIR, spec_name)
                                os.makedirs(species_dir, exist_ok=True)
                                timestamp = int(time.time() * 1000)
                                target = os.path.join(species_dir, f"{timestamp}_{os.path.basename(info.filename)}")
                                with z.open(info) as imgf, open(target, "wb") as out:
                                    out.write(imgf.read())
                                if spec_name not in st.session_state.label_manager.labels:
                                    st.session_state.label_manager.add_label(spec_name)
                                    st.session_state.database.add_species(spec_name, invasive_flag)
                else:
                    # single image file
                    if not species_name:
                        st.warning(f"Skipping {uf.name}: species name required for single images")
                        continue
                    species_dir = os.path.join(DATA_DIR, species_name)
                    os.makedirs(species_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    target = os.path.join(species_dir, f"{timestamp}_{uf.name}")
                    with open(target, "wb") as f:
                        f.write(uf.read())
            
            if species_name and species_name not in st.session_state.label_manager.labels:
                st.session_state.label_manager.add_label(species_name)
                st.session_state.database.add_species(species_name, invasive_flag)
            st.success("Images uploaded successfully!")
    
    st.write("---")
    st.subheader("Step 2: Select species to train on")
    
    # Get all available species folders
    available_species = []
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path) and len(os.listdir(item_path)) > 0:
                num_images = len([f for f in os.listdir(item_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                if num_images > 0:
                    available_species.append((item, num_images))
    
    if not available_species:
        st.info("No species with training images found. Upload some images first.")
    else:
        st.write(f"Found {len(available_species)} species with training images:")
        for sp, count in available_species:
            st.write(f"  - {sp}: {count} images")
        
        selected_species = st.multiselect(
            "Select species to train on (all selected species will be trained together)",
            [sp[0] for sp in available_species],
            default=[sp[0] for sp in available_species]
        )
        
        if selected_species:
            epochs = st.slider("Training epochs", min_value=1, max_value=20, value=5)
            if st.button("Start training on selected species"):
                st.write(f"Training model on: {', '.join(selected_species)}")

                # keep labels in sync with data directory (avoids target out-of-bounds)
                from invasive_plant_identifier.utils import sync_label_manager_with_data

                sync_label_manager_with_data(st.session_state.label_manager, DATA_DIR)

                # rebuild classifier so output layer matches new num classes
                st.session_state.classifier = PlantClassifier(
                    st.session_state.label_manager, model_path=MODEL_PATH
                )

                # build temporary directory containing only selected species
                import shutil, tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    for sp in selected_species:
                        src = os.path.join(DATA_DIR, sp)
                        dst = os.path.join(tmpdir, sp)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)

                    if val_uploaded:
                        from invasive_plant_identifier.utils import create_imagefolder_datasets_from_dirs

                        with tempfile.TemporaryDirectory() as val_dir:
                            for uf in val_uploaded:
                                if uf.name.lower().endswith(".zip"):
                                    import zipfile

                                    with zipfile.ZipFile(uf) as z:
                                        for info in z.infolist():
                                            if info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                                                parts = info.filename.split("/")
                                                if len(parts) >= 2:
                                                    spec_name = parts[0]
                                                else:
                                                    spec_name = val_species_name if val_species_name else "validation"
                                                if spec_name not in selected_species:
                                                    # ignore species not in selected training set
                                                    continue
                                                species_dir = os.path.join(val_dir, spec_name)
                                                os.makedirs(species_dir, exist_ok=True)
                                                timestamp = int(time.time() * 1000)
                                                target = os.path.join(
                                                    species_dir,
                                                    f"{timestamp}_{os.path.basename(info.filename)}",
                                                )
                                                with z.open(info) as imgf, open(target, "wb") as out:
                                                    out.write(imgf.read())
                                else:
                                    if not val_species_name:
                                        st.warning(
                                            f"Skipping {uf.name}: validation species name required for single images"
                                        )
                                        continue
                                    if val_species_name not in selected_species:
                                        st.warning(
                                            f"Skipping {uf.name}: validation species '{val_species_name}' not selected for training"
                                        )
                                        continue
                                    species_dir = os.path.join(val_dir, val_species_name)
                                    os.makedirs(species_dir, exist_ok=True)
                                    timestamp = int(time.time() * 1000)
                                    target = os.path.join(species_dir, f"{timestamp}_{uf.name}")
                                    with open(target, "wb") as f:
                                        f.write(uf.read())

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
                        st.success(f"Deleted {sp}")
                    except Exception as e:
                        st.error(f"Failed to delete {sp}: {e}")
                st.experimental_rerun()
        if st.button("Wipe all training data"):
            confirm = st.checkbox("I understand this will delete every training image", key="wipe_confirm")
            if confirm and st.button("Confirm wipe", key="wipe_confirm_btn"):
                from invasive_plant_identifier.utils import wipe_training_data
                wipe_training_data(DATA_DIR)
                st.success("All training data removed")
                st.experimental_rerun()

elif mode == "Database":
    st.header("Database")
    rows = st.session_state.database.get_all_detections()
    if not rows:
        st.info("No detections yet.")
        df = pd.DataFrame()  # empty dataframe
    else:
        # Convert sqlite3.Row objects to dict for proper column naming
        data = [dict(row) for row in rows]
        df = pd.DataFrame(data)
        # Ensure proper column order and naming
        column_order = ["id", "datetime", "analysis_time", "confidence_score", "species", "is_invasive", "image_id", "is_correct", "latitude", "longitude"]
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
                    st.session_state.database.update_detection(int(orig["id"]), **changes)
            st.success("Database updated")
        # deletion UI
        if "id" in df.columns:
            ids_to_delete = st.multiselect(
                "Select rows to delete (by id)", df["id"].tolist()
            )
            if st.button("Delete selected rows") and ids_to_delete:
                for rid in ids_to_delete:
                    st.session_state.database.delete_detection(int(rid))
                st.success("Rows deleted")
                st.rerun()
        if st.button("Export CSV"):
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            st.session_state.database.export_csv(tmpfile.name)
            st.download_button("Download data", open(tmpfile.name, "rb"), file_name="detections.csv")
    st.write("---")
    st.subheader("Species management")
    species_list = [r[0] for r in st.session_state.database.conn.execute("SELECT name FROM species").fetchall()]
    selected = st.selectbox("Select species to mark invasive/non-invasive", [""] + species_list)
    if selected:
        current = st.session_state.database.conn.execute(
            "SELECT is_invasive FROM species WHERE name = ?", (selected,)
        ).fetchone()[0]
        is_unknown = selected == "unknown"
        if is_unknown:
            st.info("'unknown' species cannot be marked as invasive. It represents low-confidence detections.")
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
            unknown_count = len(df[df["species"] == "unknown"])
            known_count = len(df[df["species"] != "unknown"])
            st.metric("Known Species Detections", known_count)
            st.metric("Unknown Species Detections", unknown_count)
        
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
            if st.button("🗑️ Clear all detections", use_container_width=True):
                st.session_state["db_wipe_pending"] = True
        
        # show confirmation only if wipe was requested
        if st.session_state.get("db_wipe_pending", False):
            st.warning("⚠️ This will permanently delete **all** detection records from the database. This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, delete all", use_container_width=True, key="confirm_wipe"):
                    st.session_state.database.clear_detections()
                    st.session_state["db_wipe_pending"] = False
                    st.success("All detections removed from database")
                    st.experimental_rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_wipe"):
                    st.session_state["db_wipe_pending"] = False
                    st.rerun()
