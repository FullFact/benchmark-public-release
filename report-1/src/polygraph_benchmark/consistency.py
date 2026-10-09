"""Helpers for analysing consistency of atomic facts across models and dates.
In general, we only want to see if the same atomic fact is extracted by the same
model on 2+ days. This is a measure of the consistency of each model's outputs.
However, if an atom is NOT a repeat of the model's past atoms, we also record if
it is a repeat of a different model's atoms. This makes factuality evaluation more
efficient. As long as we search for repeats from oldest to newest, and within-model
before between-model, then everything should work fine...
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

from api.models import Point
from api.service import MarkingSchemeService, PolygraphResponseService
from polygraph_benchmark.deduplication import check_entity_match, get_earliest_matches
from polygraph_benchmark.numbers import check_number_match

logger = logging.getLogger(__name__)


@dataclass
class DayCounts:
    """Counts of new vs repeated atoms for a single day."""

    new: int = 0
    repeat: int = 0
    total: int = 0
    new_frac: float = 0.0
    # TODO: may add a counter for repeats across different models?

    def __str__(self) -> str:
        return (
            f"DayCounts(new={self.new}, repeat={self.repeat}, total={self.total}, "
            f"new_frac={self.new_frac:.1%})"
        )


service = MarkingSchemeService()
llm_responses = PolygraphResponseService()


def get_atoms(question: str, model: str, date_from: date, date_to: date) -> list[Point]:
    """Look up & return atoms from an inclusive range of dates,
    sorted earliest first."""
    return service.get_points_for_range(question, model, date_from, date_to)


def get_existing_matches_for_question(
    question: str, day_to_process: date
) -> dict[Point, Point | None]:
    """
    Finds existing matches for the responses to the given question on the given day.
    First matches today's atoms against earlier days, then matches any still-unmatched
    today atoms against each other (earlier-time -> later-time only).
    Prioritises matches from the same model, but failing that will match across models.
    """
    day_before = day_to_process - timedelta(days=1)
    models = llm_responses.get_all_models()

    new_atoms = []
    prev_atoms = []
    for model in models:
        new_atoms += service.get_points_for_range(
            question, model, day_to_process, day_to_process
        )
        prev_atoms += service.get_points_for_range(question, model, None, day_before)

    new_atoms = [atom for atom in new_atoms if not atom.repeat_of]
    prev_atoms = [atom for atom in prev_atoms if not atom.repeat_of]

    matches: dict[Point, Point | None] = {atom: None for atom in new_atoms}

    # Pass 1: match today's atoms against previous days
    logger.info("MATCHING: Matching today's atoms against previous days.")
    if prev_atoms:
        matches.update(
            get_earliest_matches(
                new_atoms,
                prev_atoms,
                extra_filters=[check_entity_match, check_number_match],
            )
        )

    # Pass 2: still-unmatched today atoms may repeat earlier today atoms
    unmatched = [atom for atom, parent in matches.items() if parent is None]
    logger.info(
        f"MATCHING: Matching today's {len(unmatched)} "
        "unmatched atoms against each other."
    )
    if len(unmatched) >= 2:
        within_day = get_earliest_matches(
            unmatched,
            unmatched,
            extra_filters=[check_entity_match, check_number_match],
        )
        for atom, parent in within_day.items():
            if parent is not None:
                matches[atom] = parent

    return matches


def find_repeated_atoms(day_to_process: date) -> None:
    """
    Finds atoms on the given day which match previous atoms.
    """
    questions = llm_responses.get_all_questions()
    for i, question in enumerate(questions):
        start_time = time.monotonic()
        q_text: str = question["question"]
        logger.info(
            f"MATCHES: Finding matches for question {i+1}/{len(questions)}: {q_text}"
        )
        earliest_matches = get_existing_matches_for_question(q_text, day_to_process)
        for current_new, current_known in earliest_matches.items():
            if current_known:
                service.mark_as_repeat(current_new, current_known)
        end_time = time.monotonic()
        logger.info(f"MATCHES: Completed in {end_time - start_time:.2f}s")


        # else:
        #     print(
        #         f"  {point.response_time:%Y-%m-%d %H:%M} "
        #         f"{point.atomic_fact} "
        #         f"(new)"
        #     )


def group_atoms_by_repeats(
    question: str,
    model: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict[Point, list[Point]]:
    """Return a dict mapping each original atom to the list of atoms that repeat it.

    Atoms for the given question and model are fetched within [date_from, date_to].
    Atoms whose repeat_of field is set are treated as children of the atom they
    point to. Only atoms with at least one child are included as keys.

    Pass model="ALL_MODELS" to aggregate atoms across all known models.
    """
    if model == "ALL_MODELS":
        all_atoms: list[Point] = []
        for m in llm_responses.get_all_models():
            all_atoms += service.get_points_for_range(question, m, date_from, date_to)
    else:
        all_atoms = service.get_points_for_range(question, model, date_from, date_to)
    by_id = {p.id: p for p in all_atoms}
    out_of_range_parents: dict[int, Point | None] = {}

    groups: dict[Point, list[Point]] = {}
    for atom in all_atoms:
        if atom.repeat_of is not None:
            parent = by_id.get(atom.repeat_of)
            if parent is None:
                if atom.repeat_of not in out_of_range_parents:
                    # Not in the date-filtered pool - it may still exist
                    # outside the selected range. Fetch it directly so an
                    # in-range child doesn't get dropped just because its
                    # parent falls outside the filter.
                    out_of_range_parents[atom.repeat_of] = service.get_point(
                        atom.repeat_of
                    )
                parent = out_of_range_parents[atom.repeat_of]
            if parent is not None:
                groups.setdefault(parent, []).append(atom)
    return groups


def count_atoms(question: str, model: str, day: date) -> DayCounts:
    """Return new/repeat/total counts for a single (question, model, day)."""
    atoms = get_atoms(question, model, day, day)
    counts = DayCounts()
    for atom in atoms:
        if atom.repeat_of:
            parent = service.get_point(atom.repeat_of)
            if parent and parent.model == model:
                counts.repeat += 1
        else:
            counts.new += 1
        counts.total += 1
    counts.new_frac = counts.new / counts.total if counts.total else 0.0
    return counts


def consistency_score_all_responses(
    day_to_process: date,
) -> dict[str, dict[str, DayCounts]]:
    """Score all model responses for consistency for all questions on the
    given day.

    Returns a nested dict: {question_prompt: {model: DayCounts}}
    """
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()
    results: dict[str, dict[str, DayCounts]] = {}
    for question in all_questions:
        prompt = question["question"]
        results[prompt] = {}
        for model in all_models:
            try:
                results[prompt][model] = count_atoms(prompt, model, day_to_process)
            except Exception as e:
                print(f"Error scoring {model} / {prompt}: {e}")
    return results


def calculate_consistency_score(counts: DayCounts) -> float:
    """Compute a weighted overall score: combining novelty etc."""
    if counts.total == 0:
        score = 0.0
    else:
        score = 1.0 - counts.new_frac
    return score
