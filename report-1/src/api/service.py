"""Stand-in service layer for the public release.

In production, these two classes talk to the benchmark's database
(atoms, marking schemes, scores andd LLM responses).
Here, both read from the local SQLite export instead, so the
extraction, deduplication and scoring code can run unchanged.

The export is opened read-only. Anything the pipeline would write back (new
atoms, repeat links, URLs, daily scores) is kept in memory for the life of the
process. Set BENCHMARK_DB to point at a different export.
"""

import json
import logging
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from api.models import (
    DailyScore,
    MarkingScheme,
    Paragraph,
    ParagraphAnnotation,
    Point,
)
from api.ner import get_entities_batch

logger = logging.getLogger(__name__)

# The exports live in the shared data/ folder at the repository root.
DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark-2026-07-13-to-2026-09-22-filtered.sqlite"
)

# The export uses its own labels for paragraph annotations.
EXPORT_ANNOTATIONS: dict[str, ParagraphAnnotation] = {
    "essential": ParagraphAnnotation.ESSENTIAL,
    "optional": ParagraphAnnotation.CREDIT,
    "harmful": ParagraphAnnotation.HARMFUL,
    "irrelevant": ParagraphAnnotation.IRRELEVANT,
    "compound_answer": ParagraphAnnotation.COMPOUND,
    "harmful_irrelevant": ParagraphAnnotation.HARMFUL_IRRELEVANT,
    "uncategorised": ParagraphAnnotation.DIFFICULT,
}

# Atoms read from the export, with the question text and, where the atom came
# from an exported response, that response's timestamp.
ATOM_QUERY = """
    SELECT a.*, q.text AS question_text, r.created_at AS responded_at
    FROM atoms a
    JOIN questions q ON q.id = a.question_id
    LEFT JOIN responses r ON r.id = a.response_id
"""


def _naive_utc(timestamp: str) -> datetime:
    """Parse an ISO timestamp or date as a naive UTC datetime, so that atoms
    with and without a response timestamp can be compared."""
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


class _Store:
    """The read-only export plus the in-memory changes made during this run.
    Shared by every service instance, so the module-level services held by
    each pipeline stage all see the same state."""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.new_points: dict[int, Point] = {}  # atoms added during this run
        self.repeat_of: dict[int, int] = {}  # repeat links found during this run
        self.urls: dict[tuple[str, str, date], dict[str, list[str]]] = {}
        self.scores: list[DailyScore] = []
        max_id = self.conn.execute("SELECT MAX(id) FROM atoms").fetchone()[0]
        self.next_id = (max_id or 0) + 1


_stores: dict[str, _Store] = {}


def _get_store() -> _Store:
    db_path = os.environ.get("BENCHMARK_DB", DEFAULT_DB_PATH)
    if db_path not in _stores:
        _stores[db_path] = _Store(db_path)
    return _stores[db_path]


class MarkingSchemeService:
    """Atoms, marking schemes and scores.

    In production, this reads from and writes to the benchmark database."""

    def __init__(self) -> None:
        self.store = _get_store()

    def _row_to_point(self, row: sqlite3.Row) -> Point:
        response_time = _naive_utc(row["responded_at"] or row["fact_date"])
        point = Point(
            id=row["id"],
            atomic_fact=row["atomic_fact"],
            evidence=json.loads(row["evidence"]) if row["evidence"] else [],
            entities=json.loads(row["entities"]) if row["entities"] else None,
            question=row["question_text"],
            response_time=response_time,
            model=row["model"] or row["source"],
            repeat_of=row["repeat_of"],
            within_model_repeat=row["within_model_repeat"],
        )
        if point.id in self.store.repeat_of:
            point.repeat_of = self.store.repeat_of[point.id]
        return point

    def _query_points(self, where: str, params: tuple) -> list[Point]:
        rows = self.store.conn.execute(f"{ATOM_QUERY} WHERE {where}", params)
        return [self._row_to_point(row) for row in rows]

    def _new_points(
        self,
        question: str,
        model: str | None,
        date_from: date | None,
        date_to: date | None,
    ) -> list[Point]:
        return [
            p
            for p in self.store.new_points.values()
            if p.question == question
            and (model is None or p.model == model)
            and (date_from is None or p.response_time.date() >= date_from)
            and (date_to is None or p.response_time.date() <= date_to)
        ]

    def get_points_for_range(
        self, question: str, model: str, date_from: date | None, date_to: date | None
    ) -> list[Point]:
        """Return Points for question+model in [date_from, date_to], ascending by
        response_time. If either date is None, extend back (forwards) to all dates."""
        points = self._query_points(
            "q.text = ? AND a.model = ? AND a.fact_date >= ? AND a.fact_date <= ?",
            (
                question,
                model,
                (date_from or date.min).isoformat(),
                (date_to or date.max).isoformat(),
            ),
        )
        points += self._new_points(question, model, date_from, date_to)
        return sorted(points, key=lambda p: p.response_time)

    def get_points(
        self,
        question: str,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[Point]:
        """Get all Points for a given question, optionally restricted to Points
        whose response_time falls in [date_from, date_to] inclusive."""
        points = self._query_points(
            "q.text = ? AND a.fact_date >= ? AND a.fact_date <= ?",
            (
                question,
                (date_from or date.min).isoformat(),
                (date_to or date.max).isoformat(),
            ),
        )
        return points + self._new_points(question, None, date_from, date_to)

    def get_point(self, point_id: int) -> Point | None:
        """Get a single Point by its database ID."""
        if point_id in self.store.new_points:
            return self.store.new_points[point_id]
        points = self._query_points("a.id = ?", (point_id,))
        return points[0] if points else None

    def add_points(self, points: list[Point]) -> list[int]:
        """Add several Points, returning their IDs in order. Named entities are
        recognised here, in one batched pass, for any point that lacks them.

        In production, this inserts the points into the database. Here, they
        are kept in memory."""
        needs_entities = [p for p in points if p.entities is None]
        if needs_entities:
            batched = get_entities_batch([p.atomic_fact for p in needs_entities])
            for point, entities in zip(needs_entities, batched):
                point.entities = entities
        ids = []
        for point in points:
            point.id = self.store.next_id
            self.store.next_id += 1
            self.store.new_points[point.id] = point
            ids.append(point.id)
        logger.info("Stored %d new atoms in memory", len(ids))
        return ids

    def mark_as_repeat(self, newer: Point, older: Point) -> None:
        """Mark newer as a repeat of older and update newer.repeat_of in-place.

        In production, this updates the atom's row in the database. Here, the
        link is kept in memory."""
        if newer.id >= 0 and older.id >= 0:
            self.store.repeat_of[newer.id] = older.id
            newer.repeat_of = older.id

    def get_all_paragraphs(self, question: str) -> list[int]:
        """Return all paragraph IDs for a given question."""
        rows = self.store.conn.execute(
            "SELECT p.id FROM paragraphs p JOIN questions q ON q.id = p.question_id "
            "WHERE q.text = ? ORDER BY p.sort_order",
            (question,),
        )
        return [row["id"] for row in rows]

    def get_paragraph(self, question: str, paragraph_id: int) -> Paragraph | None:
        """Get a Paragraph by its question and ID."""
        row = self.store.conn.execute(
            "SELECT p.* FROM paragraphs p JOIN questions q ON q.id = p.question_id "
            "WHERE q.text = ? AND p.id = ?",
            (question, paragraph_id),
        ).fetchone()
        if row is None:
            return None
        points = self._query_points(
            "a.id IN (SELECT atom_id FROM paragraph_atoms WHERE paragraph_id = ?)",
            (paragraph_id,),
        )
        return Paragraph(
            points=points,
            annotation=EXPORT_ANNOTATIONS[row["annotation"]],
            description=row["description"],
            applies_from=row["applies_from"],
            applies_to=row["applies_to"],
        )

    def get_marking_scheme(self, question: str) -> MarkingScheme | None:
        """Get the marking scheme for a question."""
        return MarkingScheme(
            question=question, paragraph_ids=self.get_all_paragraphs(question)
        )

    def create_or_get_marking_scheme(self, question: str) -> MarkingScheme:
        """In production, this creates an empty marking scheme if the question
        doesn't have one yet. Here, every question has one in the export."""
        return MarkingScheme(
            question=question, paragraph_ids=self.get_all_paragraphs(question)
        )

    def store_response_urls(
        self,
        question: str,
        model: str,
        response_date: date,
        netloc_urls: dict[str, list[str]],
    ) -> int:
        """Store URLs for a (question, model, date). Returns count stored.

        In production, this writes to the database. Here, the URLs are kept
        in memory."""
        self.store.urls[(question, model, response_date)] = netloc_urls
        return sum(len(urls) for urls in netloc_urls.values())

    def get_response_urls(
        self, question: str, model: str, response_date: date
    ) -> dict[str, list[str]]:
        """Get URLs for a (question, model, date) grouped by netloc."""
        if (question, model, response_date) in self.store.urls:
            return self.store.urls[(question, model, response_date)]
        rows = self.store.conn.execute(
            "SELECT u.netloc, u.url FROM response_urls u "
            "JOIN responses r ON r.id = u.response_id "
            "JOIN questions q ON q.id = r.question_id "
            "WHERE q.text = ? AND r.model = ? AND r.response_date = ? "
            "ORDER BY u.id",
            (question, model, response_date.isoformat()),
        )
        netloc_urls: dict[str, list[str]] = {}
        for row in rows:
            netloc_urls.setdefault(row["netloc"], []).append(row["url"])
        return netloc_urls

    def store_daily_score(self, score: DailyScore) -> None:
        """In production, this upserts a row in the daily_scores table. Here,
        the score is added to an in-memory list (see get_stored_scores)."""
        self.store.scores.append(score)

    def get_stored_scores(self) -> list[DailyScore]:
        """Return the scores stored during this run."""
        return self.store.scores

    def close(self) -> None:
        """In production, this releases the database connection between
        pipeline stages. Nothing to do here."""


class PolygraphResponseService:
    """Read-only access to the archived LLM responses.

    In production, this calls the llm-polygraph API."""

    def __init__(self) -> None:
        self.store = _get_store()

    def get_all_questions(self) -> list[dict]:
        """Return every benchmark question, as {"question": text, "tags": [...]}."""
        rows = self.store.conn.execute(
            "SELECT q.text, GROUP_CONCAT(t.name) AS tags FROM questions q "
            "LEFT JOIN question_tags qt ON qt.question_id = q.id "
            "LEFT JOIN tags t ON t.id = qt.tag_id "
            "GROUP BY q.id ORDER BY q.id"
        )
        return [
            {
                "question": row["text"],
                "tags": row["tags"].split(",") if row["tags"] else [],
            }
            for row in rows
        ]

    def get_all_models(self) -> list[str]:
        """Return all LLMs with responses in the export."""
        rows = self.store.conn.execute(
            "SELECT DISTINCT model FROM responses ORDER BY model"
        )
        return [row["model"] for row in rows]

    def get_active_db(self) -> str:
        """In production, this reports which database llm-polygraph is reading.
        Only used in log messages."""
        return "sqlite-export"

    def get_response(self, model: str, prompt: str, selected_date: date) -> dict:
        """Return the response for a given model, prompt and date, or an empty
        dict if there isn't one (or its text wasn't retained in the export)."""
        row = self.store.conn.execute(
            "SELECT r.* FROM responses r JOIN questions q ON q.id = r.question_id "
            "WHERE q.text = ? AND r.model = ? AND r.response_date = ?",
            (prompt, model, selected_date.isoformat()),
        ).fetchone()
        if row is None or not row["response_text"]:
            return dict()
        return {
            "prompt": prompt,
            "response": row["response_text"],
            "model": row["model"],
            "responded_at": _naive_utc(row["created_at"]).isoformat(),
            "created_at": _naive_utc(row["created_at"]).isoformat(),
        }
