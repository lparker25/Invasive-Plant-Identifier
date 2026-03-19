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
    assert row["run_id"] == db.get_latest_run_id()

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


def test_run_creation_and_filtering(tmp_path):
    db_path = tmp_path / "test_runs.db"
    db = Database(str(db_path))

    run1_id, run1_label = db.create_run(source_type="uploaded")
    run2_id, run2_label = db.create_run(source_type="uploaded")

    assert run1_label == "Run 1"
    assert run2_label == "Run 2"
    assert run2_id > run1_id

    db.log_detection(
        datetime="2026-03-19T10:00:00",
        analysis_time=0.2,
        confidence_score=0.8,
        species="A",
        is_invasive=False,
        latitude="N/A",
        longitude="N/A",
        run_id=run1_id,
    )
    db.log_detection(
        datetime="2026-03-19T10:01:00",
        analysis_time=0.3,
        confidence_score=0.9,
        species="B",
        is_invasive=True,
        latitude="N/A",
        longitude="N/A",
        run_id=run2_id,
    )

    run1_rows = db.get_all_detections(run_id=run1_id)
    run2_rows = db.get_all_detections(run_id=run2_id)
    assert len(run1_rows) == 1
    assert len(run2_rows) == 1
    assert run1_rows[0]["species"] == "A"
    assert run2_rows[0]["species"] == "B"

    counts_run1 = db.get_species_counts(run_id=run1_id)
    counts_run2 = db.get_species_counts(run_id=run2_id)
    assert counts_run1 == [("A", 1, 0)]
    assert counts_run2 == [("B", 1, 1)]


def test_run_scoped_clear_and_export(tmp_path):
    db_path = tmp_path / "test_run_clear.db"
    db = Database(str(db_path))

    run1_id, _ = db.create_run(source_type="uploaded")
    run2_id, _ = db.create_run(source_type="uploaded")

    db.log_detection(
        datetime="2026-03-19T11:00:00",
        analysis_time=0.15,
        confidence_score=0.85,
        species="A",
        is_invasive=False,
        latitude="N/A",
        longitude="N/A",
        run_id=run1_id,
    )
    db.log_detection(
        datetime="2026-03-19T11:01:00",
        analysis_time=0.25,
        confidence_score=0.95,
        species="B",
        is_invasive=True,
        latitude="N/A",
        longitude="N/A",
        run_id=run2_id,
    )

    export_path = tmp_path / "run1.csv"
    db.export_csv(str(export_path), run_id=run1_id)
    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    assert "A" in content
    assert "B" not in content

    db.clear_detections(run_id=run1_id)
    assert db.get_all_detections(run_id=run1_id) == []
    assert len(db.get_all_detections(run_id=run2_id)) == 1

