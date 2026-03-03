import os
import tempfile

import pytest

from invasive_plant_identifier.db import Database


def test_database_crud(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    # initially empty
    rows = db.get_all_detections()
    assert rows == []

    # add a species and a detection
    db.add_species("TestPlant", is_invasive=True)
    db.log_detection(
        datetime="2026-03-02T12:00:00",
        analysis_time=0.1,
        confidence_score=0.99,
        species="TestPlant",
        is_invasive=True,
        latitude="N/A",
        longitude="N/A",
    )
    rows = db.get_all_detections()
    assert len(rows) == 1
    row = rows[0]
    assert row["species"] == "TestPlant"
    assert row["is_invasive"] == 1

    # update entry
    db.update_detection(row["id"], species="OtherPlant")
    updated = db.get_all_detections()[0]
    assert updated["species"] == "OtherPlant"

    # delete entry
    db.delete_detection(updated["id"])
    assert db.get_all_detections() == []

    # export (should create file without errors)
    out_csv = tmp_path / "out.csv"
    db.log_detection(
        datetime="2026-03-02T12:00:00",
        analysis_time=0.1,
        confidence_score=0.99,
        species="TestPlant2",
        is_invasive=False,
        latitude="N/A",
        longitude="N/A",
    )
    db.export_csv(str(out_csv))
    assert out_csv.exists()

    # clear all rows and verify
    db.clear_detections()
    assert db.get_all_detections() == []

    # test image_id logging
    db.log_detection(
        datetime="2026-03-02T12:00:00",
        analysis_time=0.1,
        confidence_score=0.95,
        species="TestPlant",
        is_invasive=True,
        latitude="N/A",
        longitude="N/A",
        image_id="test_image.jpg",
    )
    rows = db.get_all_detections()
    assert len(rows) == 1
    assert rows[0]["image_id"] == "test_image.jpg"
    # new is_correct flag should default to true (1)
    assert rows[0]["is_correct"] == 1
    # updating the flag works
    db.update_detection(rows[0]["id"], is_correct=0)
    assert db.get_all_detections()[0]["is_correct"] == 0

