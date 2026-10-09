"""Main entry point: get_earliest_matches()"""

import logging
import re
from collections.abc import Callable
from datetime import datetime

import torch
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch import Tensor

from api.models import Entity, EntityType, Point
from api.service import MarkingSchemeService

logger = logging.getLogger(__name__)
sentence_transformer = None
cross_encoder = None
service = None

ENTITY_OVERLAP_THRESHOLD: float = 1.0


def get_service() -> MarkingSchemeService:
    """Initialise connection to API/db only when required"""
    global service
    if not service:
        service = MarkingSchemeService()
    return service


def get_sentence_transformer() -> SentenceTransformer:
    global sentence_transformer
    if not sentence_transformer:
        sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return sentence_transformer


def get_cross_encoder() -> CrossEncoder:
    global cross_encoder
    if not cross_encoder:
        cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    return cross_encoder


def encode_points(points: list[Point]) -> Tensor:
    model = get_sentence_transformer()
    text = [p.atomic_fact for p in points]
    encoded = model.encode(
        text, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    return encoded


def quick_comparison(
    new_points: list[Point], known_points: list[Point]
) -> list[tuple[Point, Point]]:
    """
    Given a two lists of points, return suspected matches.
    Uses a quick vector comparison method.
    """
    logger.info(
        f"DEDUPE: Calculating similarities for {len(new_points)} new "
        f"Points against {len(known_points)} known Points."
    )
    if not new_points or not known_points:
        return []
    new_emb = encode_points(new_points)
    known_emb = encode_points(known_points)
    scores = util.dot_score(new_emb, known_emb)

    threshold = 0.7
    match_new_idx, match_known_idx = torch.where(scores >= threshold)
    pairs = [
        (new_points[i], known_points[j]) for i, j in zip(match_new_idx, match_known_idx)
    ]
    logger.info(f"DEDUPE: Found {len(pairs)} potential matches.")
    return pairs


def accurate_comparison(
    point_pairs: list[tuple[Point, Point]],
) -> list[tuple[Point, Point]]:
    """
    Given a list of Point pairs, return suspected matches.
    Uses a slow but powerful cross-encoding method.
    """
    logger.info(
        f"DEDUPE: Checking which of the {len(point_pairs)} "
        "potential matches are matches"
    )
    model = get_cross_encoder()
    sentence_pairs = [
        (new.atomic_fact, known.atomic_fact) for new, known in point_pairs
    ]
    scores = model.predict(sentence_pairs, show_progress_bar=False)

    threshold = 0.7
    matching_pairs = [
        pair for pair, score in zip(point_pairs, scores) if score >= threshold
    ]
    logger.info(f"DEDUPE: Found {len(matching_pairs)} matches")
    return matching_pairs


def filter_one_match_per_atom(
    matching_pairs: list[tuple[Point, Point]],
) -> dict[Point, Point]:
    """
    Given a list of pairs, finds the earliest match.
    Prioritises within-model matches over inter-model matches.
    So if Gemini has said the thing before,
    that will be chosen over an earlier repeat by Grok.
    """

    def rank(new: Point, known: Point) -> tuple[bool, datetime]:
        """
        This function makes a tuple of
        1. Whether the models match
        2. The response time

        If this tuple is lower for a new point than the currently stored one
        then it should be kept instead.
        """
        return (known.model != new.model, known.response_time)

    earliest_matches: dict[Point, Point] = {}
    for new, known in matching_pairs:
        if known.response_time > new.response_time or known.id == new.id:
            continue
        current_earliest = earliest_matches.get(new)
        if current_earliest is None or rank(new, known) < rank(new, current_earliest):
            earliest_matches[new] = known

    return earliest_matches


def run_filter_functions(
    matches: list[tuple[Point, Point]],
    filters: list[Callable[[Point, Point], bool]] | None,
) -> list[tuple[Point, Point]]:
    """
    Runs a list of filter functions on Point pairs.
    """
    filtered_matches = matches
    for filter_func in filters or []:
        current_filtered_matches = []
        filter_failed_count = 0
        for new, known in filtered_matches:
            try:
                if filter_func(new, known):
                    current_filtered_matches.append((new, known))
            except ValueError:
                filter_failed_count += 1
                current_filtered_matches.append((new, known))
        if filter_failed_count:
            logger.info(
                f"Filter {filter_func.__name__} failed for {filter_failed_count} pairs."
            )
        filtered_matches = current_filtered_matches

    return filtered_matches


def get_earliest_matches(
    new_points: list[Point],
    known_points: list[Point],
    extra_filters: list[Callable[[Point, Point], bool]] | None = None,
) -> dict[Point, Point | None]:
    """
    Finds earliest matching known point for each provided point.
    Does a filter across all pairs with a quick model (pairwise cosine
    similarity). Takes any pairs whose text is near-identical as matches from
    there, skipping the filters and the more thorough check. Then applies any
    specified filters to the remaining pairs. Then does a more thorough check
    on those with a more complex cross-encoding model. Finally filters down
    to only keep one match per atom.
    """
    if not new_points or not known_points:
        return {point: None for point in new_points}

    # run an initial quick comparison
    initial_matches = quick_comparison(new_points, known_points)
    if not initial_matches:
        return {point: None for point in new_points}

    # pairs that are identical bar punctuation need no further checking
    near_identical: list[tuple[Point, Point]] = []
    remaining_matches: list[tuple[Point, Point]] = []
    for pair in initial_matches:
        target = near_identical if near_identical_text(*pair) else remaining_matches
        target.append(pair)
    logger.info(
        f"DEDUPE: {len(near_identical)} of the potential matches are near-identical."
    )

    # run the extra filters on the rest
    final_matches = near_identical
    filtered_matches = run_filter_functions(remaining_matches, extra_filters)

    # run a second, more accurate comparison on those matches
    if filtered_matches:
        final_matches = final_matches + accurate_comparison(filtered_matches)
    if not final_matches:
        return {point: None for point in new_points}

    # keep only one match per new atom
    earliest_matches = filter_one_match_per_atom(final_matches)

    logger.info(
        f"DEDUPE: {len(earliest_matches)} out of {len(new_points)} points have a match."
    )
    return {point: earliest_matches.get(point) for point in new_points}


def fuzzy_match(string1: str, string2: str) -> bool:
    """
    Match two strings by fuzzy substring similarity.

    They match if both of the following hold:

    1. they share a substring with a fuzzy similarity of at least 90, and
    2. that substring is at least 40% of each label.

    This allows annotations worded slightly differently -- e.g. a surname used
    on its own instead of the full name.
    """
    string1 = string1.lower().strip()
    string2 = string2.lower().strip()
    similarity = fuzz.partial_ratio_alignment(string1, string2)
    if not similarity:
        return False
    overlap_size = similarity.src_end - similarity.src_start
    return (
        overlap_size >= len(string1) * 0.4
        and overlap_size >= len(string2) * 0.4
        and similarity.score >= 90
    )


def _is_in_list(string: str, candidates: list[str]) -> bool:
    return any(fuzzy_match(string, candidate) for candidate in candidates)


def get_entity_overlap(new_ents: list[Entity], og_ents: list[Entity]) -> float:
    """
    Calculates the proportion of entities in `og_ents` that are contained
    within `new_ents`.
    1.0 would mean all of the original entities are contained.
    0.0 would mean none of the original entities are contained.

    Numbers are ignored as they will be compared separately.
    """
    og_ent_text: list[str] = [
        e.text for e in og_ents if e.ent_type != EntityType.NUMBER
    ]
    new_ent_text: list[str] = [
        e.text for e in new_ents if e.ent_type != EntityType.NUMBER
    ]

    # No non-number entities to disagree on, so treat as a full overlap.
    if not og_ent_text:
        return 1.0

    matches = [_is_in_list(ent, new_ent_text) for ent in og_ent_text]
    return sum(matches) / len(og_ent_text)


def check_entity_match(new_atom: Point, og_atom: Point) -> bool:
    """
    Checks whether a new atom contains the same entities as an original atom,
    above the specified threshold `ENTITY_OVERLAP_THRESHOLD`.
    """
    if og_atom.entities is None or new_atom.entities is None:
        raise ValueError(
            "One of the atoms has a value of `None` for entities. "
            "Cannot compare overlap for `None` values."
        )
    entity_overlap = get_entity_overlap(new_atom.entities, og_atom.entities)
    return entity_overlap >= ENTITY_OVERLAP_THRESHOLD


def normalise_for_comparison(text: str) -> str:
    """
    Lowercases text and replaces punctuation with whitespace, so that atoms
    differing only in punctuation, casing or spacing compare as identical.

    Thousands separators and apostrophes are dropped, so that "900,000" agrees
    with "900000" and "the NHS's budget" with "the NHSs budget". Other
    punctuation becomes a space rather than being dropped, so that "4.2" does
    not collapse into "42".
    """
    # TODO: Consider compiling these regexes to save time
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.casefold())
    text = re.sub(r"['’ʼ]", "", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def near_identical_text(new_atom: Point, og_atom: Point) -> bool:
    """
    Checks whether two atoms are the same words in the same order, ignoring
    differences in casing, punctuation, spacing and thousands separators.

    The comparison is exact rather than fuzzy, because atoms that pass it are
    matched without any further checking. One different word can reverse an
    atom's meaning ("£4 million" against "£4 billion") while changing very
    little of the text.
    """
    return normalise_for_comparison(new_atom.atomic_fact) == normalise_for_comparison(
        og_atom.atomic_fact
    )
