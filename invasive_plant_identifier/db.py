import csv
import os
import sqlite3
from datetime import datetime, timezone
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
    run_id INTEGER DEFAULT 1,
    FOREIGN KEY(species) REFERENCES species(name)
);

CREATE TABLE IF NOT EXISTS identification_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label TEXT UNIQUE,
    created_at TEXT,
    source_type TEXT DEFAULT 'uploaded'
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
        # backward compatibility: add run_id if it doesn't exist
        try:
            cursor.execute("ALTER TABLE detections ADD COLUMN run_id INTEGER DEFAULT 1")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

        self._ensure_legacy_run()

    def _ensure_legacy_run(self) -> None:
        """Ensure historical detections are associated with a default run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO identification_runs (run_id, run_label, created_at, source_type) VALUES (?, ?, ?, ?)",
            (1, "Legacy Run", datetime.now(timezone.utc).isoformat(), "legacy"),
        )
        cursor.execute("UPDATE detections SET run_id = 1 WHERE run_id IS NULL")
        self.conn.commit()

    def _next_run_number(self) -> int:
        """Return the next integer suffix for run labels like 'Run X'."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT run_label FROM identification_runs WHERE run_label LIKE 'Run %'"
        )
        max_num = 0
        for row in cursor.fetchall():
            label = row[0] or ""
            parts = label.split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                max_num = max(max_num, int(parts[1]))
            except ValueError:
                continue
        return max_num + 1

    def create_run(self, run_label: Optional[str] = None, source_type: str = "uploaded") -> Tuple[int, str]:
        """Create a new identification run and return (run_id, run_label)."""
        cursor = self.conn.cursor()
        if not run_label:
            run_label = f"Run {self._next_run_number()}"
        created_at = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO identification_runs (run_label, created_at, source_type) VALUES (?, ?, ?)",
            (run_label, created_at, source_type),
        )
        self.conn.commit()
        return int(cursor.lastrowid), run_label

    def list_runs(self) -> List[sqlite3.Row]:
        """List identification runs in newest-first order."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT run_id, run_label, created_at, source_type FROM identification_runs ORDER BY run_id DESC"
        )
        return cursor.fetchall()

    def get_latest_run_id(self) -> int:
        """Return latest run id; defaults to legacy run id 1."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(run_id), 1) FROM identification_runs")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] else 1

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
        run_id: Optional[int] = None,
    ) -> None:
        if run_id is None:
            run_id = self.get_latest_run_id()
        self.add_species(species, is_invasive)
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO detections (datetime, analysis_time, confidence_score, species, is_invasive, latitude, longitude, image_id, is_correct, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                run_id,
            ),
        )
        self.conn.commit()

    def get_all_detections(self, run_id: Optional[int] = None) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        if run_id is None:
            cursor.execute("SELECT * FROM detections ORDER BY datetime DESC")
        else:
            cursor.execute(
                "SELECT * FROM detections WHERE run_id = ? ORDER BY datetime DESC",
                (run_id,),
            )
        return cursor.fetchall()

    def update_detection(self, detection_id: int, run_id: Optional[int] = None, **fields) -> None:
        if not fields:
            return
        keys = ", ".join([f"{k} = ?" for k in fields.keys()])
        values = list(fields.values())
        cursor = self.conn.cursor()
        if run_id is None:
            values.append(detection_id)
            cursor.execute(f"UPDATE detections SET {keys} WHERE id = ?", values)
        else:
            values.extend([detection_id, run_id])
            cursor.execute(f"UPDATE detections SET {keys} WHERE id = ? AND run_id = ?", values)
        self.conn.commit()

    def delete_detection(self, detection_id: int, run_id: Optional[int] = None) -> None:
        cursor = self.conn.cursor()
        if run_id is None:
            cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        else:
            cursor.execute("DELETE FROM detections WHERE id = ? AND run_id = ?", (detection_id, run_id))
        self.conn.commit()

    def export_csv(self, output_path: str, run_id: Optional[int] = None) -> None:
        rows = self.get_all_detections(run_id=run_id)
        if not rows:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(list(row))

    def clear_detections(self, run_id: Optional[int] = None) -> None:
        """Delete all rows from detections, optionally scoped to a run."""
        cursor = self.conn.cursor()
        if run_id is None:
            cursor.execute("DELETE FROM detections")
        else:
            cursor.execute("DELETE FROM detections WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def get_species_counts(self, run_id: Optional[int] = None) -> List[Tuple[str, int, int]]:
        """Return list of (species, total_count, invasive_count), optionally filtered by run."""
        cursor = self.conn.cursor()
        if run_id is None:
            cursor.execute(
                "SELECT species, COUNT(*) as cnt, SUM(is_invasive) as inv "
                "FROM detections GROUP BY species"
            )
        else:
            cursor.execute(
                "SELECT species, COUNT(*) as cnt, SUM(is_invasive) as inv "
                "FROM detections WHERE run_id = ? GROUP BY species",
                (run_id,),
            )
        return [(r[0], r[1], r[2] or 0) for r in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
