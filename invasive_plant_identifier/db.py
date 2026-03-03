import csv
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS species (
    name TEXT PRIMARY KEY,
    is_invasive INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT,
    analysis_time REAL,
    confidence_score REAL,
    species TEXT,
    is_invasive INTEGER,
    latitude TEXT,
    longitude TEXT,
    image_id TEXT,
    is_correct INTEGER DEFAULT 1,
    FOREIGN KEY(species) REFERENCES species(name)
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        cursor = self.conn.cursor()
        cursor.executescript(DB_SCHEMA)
        # backward compatibility: add image_id column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE detections ADD COLUMN image_id TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        # backward compatibility: add is_correct flag
        try:
            cursor.execute("ALTER TABLE detections ADD COLUMN is_correct INTEGER DEFAULT 1")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def add_species(self, name: str, is_invasive: bool = False) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO species (name, is_invasive) VALUES (?, ?)",
            (name, 1 if is_invasive else 0),
        )
        self.conn.commit()

    def set_invasive(self, name: str, is_invasive: bool) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE species SET is_invasive = ? WHERE name = ?",
            (1 if is_invasive else 0, name),
        )
        # update existing detection rows to keep them in sync
        cursor.execute(
            "UPDATE detections SET is_invasive = ? WHERE species = ?",
            (1 if is_invasive else 0, name),
        )
        self.conn.commit()

    def log_detection(
        self,
        datetime: str,
        analysis_time: float,
        confidence_score: float,
        species: str,
        is_invasive: bool,
        latitude: Any,
        longitude: Any,
        image_id: str = "N/A",
        is_correct: Optional[bool] = True,
    ) -> None:
        self.add_species(species, is_invasive)
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO detections (datetime, analysis_time, confidence_score, species, is_invasive, latitude, longitude, image_id, is_correct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime,
                analysis_time,
                confidence_score,
                species,
                1 if is_invasive else 0,
                latitude,
                longitude,
                image_id,
                1 if is_correct else 0,
            ),
        )
        self.conn.commit()

    def get_all_detections(self) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY datetime DESC")
        return cursor.fetchall()

    def update_detection(self, detection_id: int, **fields) -> None:
        if not fields:
            return
        keys = ", ".join([f"{k} = ?" for k in fields.keys()])
        values = list(fields.values())
        values.append(detection_id)
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE detections SET {keys} WHERE id = ?", values)
        self.conn.commit()

    def delete_detection(self, detection_id: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        self.conn.commit()

    def export_csv(self, output_path: str) -> None:
        rows = self.get_all_detections()
        if not rows:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(list(row))

    def clear_detections(self) -> None:
        """Delete all rows from the detections table."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM detections")
        self.conn.commit()

    def get_species_counts(self) -> List[Tuple[str, int, int]]:
        """Return list of (species, total_count, invasive_count)"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT species, COUNT(*) as cnt, SUM(is_invasive) as inv "
            "FROM detections GROUP BY species"
        )
        return [(r[0], r[1], r[2] or 0) for r in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
