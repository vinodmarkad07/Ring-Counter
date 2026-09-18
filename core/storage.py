"""Small SQLite history store for local and single-instance deployments."""

from __future__ import annotations

import csv
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, path: Path, limit: int = 100) -> None:
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    ring_count INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    sharpness REAL NOT NULL,
                    brightness REAL NOT NULL,
                    glare REAL NOT NULL,
                    consistency REAL NOT NULL
                )
                """
            )

    def add(self, image_name: str, result: dict) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO counts
                (created_at, image_name, ring_count, confidence, confidence_score,
                 processing_time, sharpness, brightness, glare, consistency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    image_name[:255],
                    int(result["ring_count"]),
                    result["confidence"],
                    float(result["confidence_score"]),
                    float(result["processing_time"]),
                    float(result["sharpness"]),
                    float(result["brightness"]),
                    float(result["glare"]),
                    float(result["consistency"]),
                ),
            )
            db.execute(
                "DELETE FROM counts WHERE id NOT IN (SELECT id FROM counts ORDER BY id DESC LIMIT ?)",
                (self.limit,),
            )

    def list(self, limit: int = 25) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM counts ORDER BY id DESC LIMIT ?", (max(1, min(limit, self.limit)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, record_id: int) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM counts WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    def csv_bytes(self) -> bytes:
        output = io.StringIO()
        rows = self.list(self.limit)
        fields = [
            "id", "created_at", "image_name", "ring_count", "confidence",
            "confidence_score", "processing_time", "sharpness", "brightness",
            "glare", "consistency",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8")