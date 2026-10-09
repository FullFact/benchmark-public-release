"""Data models for marking schemes, paragraphs, and points."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class EntityType(StrEnum):
    """
    The type of entity, e.g. person, organisation.

    You can easily add more by adding more options to the enum.
    The model does not need to be retrained.
    """

    PERSON = "person"
    LOCATION = "location"
    ORGANISATION = "organisation"
    NUMBER = "number"


class Entity(BaseModel):
    text: str
    ent_type: EntityType
    start: int
    end: int
    score: float


class Point(BaseModel):
    """Store points: a statement with evidence (typically a URL)."""

    model_config = ConfigDict(frozen=False)

    atomic_fact: str  # minimal version of a statement
    evidence: list[str]  # E.g. URLs supporting a fact
    entities: list[Entity] | None = None
    question: str
    response_time: datetime
    model: str
    repeat_of: int | None = None  # ID of point this duplicates, if any
    within_model_repeat: int | None = None  # ID of earliest same-model match
    id: int = -1  # primary key for db (-1 = not yet written)

    # Pydantic v2 sets __hash__ = None on mutable models; restore it explicitly.
    def __hash__(self) -> int:
        return hash((self.atomic_fact, self.question))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return (self.atomic_fact, self.question) == (other.atomic_fact, other.question)


class Paragraph(BaseModel):
    """An idealised paragraph will contain a number of Points"""

    points: list[Point]
    annotation: ParagraphAnnotation
    description: str | None = None
    applies_from: date | None = None
    applies_to: date | None = None


class MarkingScheme(BaseModel):
    """A question text with a list of paragraph ids"""

    question: str
    paragraph_ids: list[int]


AGGREGATE_QUESTION = "AGGREGATE"  # sentinal value


class CivicsScore(TypedDict, total=False):
    civics_mean: float
    non_civics_mean: float
    overall_score: float


class TransparencyScore(TypedDict, total=False):
    transparency_mean: float
    non_transparency_mean: float
    overall_score: float


class DailyScore(BaseModel):
    """Scores for one question/model/date combination.
    Data persisted to daily_scores table."""

    response_date: str
    question: str
    model: str
    factual: dict[str, int | float]
    consistency: dict[str, int | float]
    civics: CivicsScore  # should default to {} ?
    timeliness: float
    transparency: TransparencyScore


class ParagraphAnnotation(StrEnum):
    """Define all possible annotations for a paragraph.
    The description field is used in the UI, the value is a key
    used in the database"""

    description: str

    def __new__(cls, value: str, description: str = "") -> "ParagraphAnnotation":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    # Note: Some of these labels are over-written in the UI:
    # See cloudrun_sql.main.py / ANNOTATION_LABELS
    ESSENTIAL = "Essential", "Anything necessary to answering the question"
    CREDIT = "Extra credit", "Useful, relevant but not essential"
    HARMFUL = "Harmful", "Inaccurate, misleading, shaky or unsubstantiated"
    IRRELEVANT = "Irrelevant", "Not essential or harmful or worth extra credit"
    DIFFICULT = (
        "Difficult to categorise",
        "A bin for atoms which were confusing or exceptional in some way, warranting "
        "further discussion or refining the categories",
    )
    COMPOUND = "Compound answer", "Atoms that depend on each other for context"
    HARMFUL_IRRELEVANT = "Harmful and irrelevant", "Wrong and off-topic"

    # Old categories:
    # ESSENTIAL = "essential", "Needed to answer question"
    # OPTIONAL = "optional", "Relevant to answer, but no penalty for missing"
    # ANY_ONE = "any one", "Any one of the atoms in this set is sufficient"
    # DIRECT_ANSWER = "direct", "A direct answer to question"
    # IMPLIED_INTENT="implied","Answer address the implicit reason behind the question"
    # CONTEXT_ONLY = "context", "Provides useful, relevant context but not essential"
    # CAVEAT = "caveat","Something user should know to interpret the answers"
    # BAD_SOURCE="unreliable evidence","Source is unreliable, even if the atom is true"
    # MISLEADING = "misleading", "False or clearly misleading atoms"
