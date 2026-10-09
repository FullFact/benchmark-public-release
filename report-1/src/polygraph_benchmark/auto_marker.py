"""Use genAI to analyse answers and compare to model answers
Generates a score reflecting the factuality of a response.
WORK IN PROGRESS!"""

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from google import genai
from google.genai import types
from pydantic import BaseModel
# from tenacity import retry, stop_after_attempt, wait_exponential

from api.models import (
    CivicsScore,
    MarkingScheme,
    Paragraph,
    ParagraphAnnotation,
    Point,
    TransparencyScore,
)
from api.service import MarkingSchemeService, PolygraphResponseService
from polygraph_benchmark.prompts import (
    ATOMIC_FACTS_EXTRACTION_PROMPT,
    ATOMIC_FACTS_EXTRACTION_PROMPT_UPDATED,
)

logger = logging.getLogger(__name__)

service = MarkingSchemeService()
llm_responses = PolygraphResponseService()


ANNOTATION_WEIGHTS: dict[ParagraphAnnotation, float] = {
    ParagraphAnnotation.ESSENTIAL: 5.0,
    ParagraphAnnotation.CREDIT: 3.0,
    ParagraphAnnotation.HARMFUL: -5.0,
    ParagraphAnnotation.IRRELEVANT: 0.0,
    ParagraphAnnotation.DIFFICULT: 0.0,
    ParagraphAnnotation.COMPOUND: 5.0,
    # Wrong and off-topic: penalised, but less than squarely harmful content,
    # which at least purports to answer the question.
    ParagraphAnnotation.HARMFUL_IRRELEVANT: -2.5,
}
ANY_ONE_WEIGHT = 3.0


@dataclass
class ModelScores:
    """Counts of atomic facts found for each paragraph annotation, plus unmatched facts.
    Also stores the maximum possible score for the marking scheme, which is used
    later to normalise these scores."""

    counts: dict[str, float] = field(
        default_factory=lambda: {
            **{a.value: 0 for a in ParagraphAnnotation},
            "missed_essential": 0.0,
            "unlabelled": 0.0,
            "unseen": 0.0,
            "basic_score": 0.0,
            "max_score": 0.0,
        }
    )


class ExtractedFact(BaseModel):
    atomic_fact: str
    url: str | None


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    logger.warning(
        "extract_atomic_facts attempt %d failed: %s: %s",
        retry_state.attempt_number,
        type(exc).__name__,
        exc,
    )


def extract_atomic_facts(response: dict) -> list[Point]:
    """An 'atomic fact' is the simplest expression of a fact. A sentence
    typically contains several atomic facts.
    """
    question = response["prompt"]
    response_text = response["response"]
    response_time = datetime.fromisoformat(
        response["responded_at"] or response["created_at"]
    )

    prompt = ATOMIC_FACTS_EXTRACTION_PROMPT
    prompt += (
        f"The question is ```{question}``` and the provided answer is:\n{response_text}"
    )

    client = genai.Client()

    marker_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[ExtractedFact],
        ),
    )

    model = response.get("model", "")
    return [
        Point(
            atomic_fact=item["atomic_fact"],
            evidence=[item["url"]] if item["url"] else [],
            question=question,
            response_time=response_time,
            model=model,
        )
        for item in json.loads(marker_response.text)
    ]


def _build_fact_to_marking_para(
    question_prompt: str, marking_scheme: MarkingScheme
) -> tuple[dict[str, int], dict[int, Paragraph]]:
    """Map each known atomic-fact text to its Paragraph id for the given question,
    and each Paragraph id to the Paragraph itself.
    Each Paragraph is a list of points and an annotation (e.g. essential, harmful).

    Paragraphs are keyed by their database id rather than by the Paragraph object,
    so that two ids can never collapse into one if get_paragraph() is ever cached.
    """
    fact_to_para_id: dict[str, int] = {}
    paras_by_id: dict[int, Paragraph] = {}
    for para_id in sorted(marking_scheme.paragraph_ids):
        para = service.get_paragraph(question_prompt, para_id)
        if para:
            paras_by_id[para_id] = para
            for pt in para.points:
                fact_to_para_id.setdefault(pt.atomic_fact, para_id)
    return fact_to_para_id, paras_by_id


def calculate_question_score(
    marks: ModelScores, outside_time_range: bool = False
) -> float:
    """Compute a weighted overall score by combining essential, optional etc.
    If outside_time_range, then penalise any matches by making the paragraph weights
    negative. This is for atoms that assert claims that are no longer true
    (even if they once were)."""
    # Defined as 'abstract Score' in 'benchmark metrics' PDF
    if outside_time_range:
        # Penalise out-of-date information
        weights: dict[ParagraphAnnotation, float] = {
            ann: -abs(w) for ann, w in ANNOTATION_WEIGHTS.items()
        }
    else:
        weights = ANNOTATION_WEIGHTS
    score = sum(
        w * marks.counts[ann] for ann, w in weights.items() if ann in marks.counts
    )
    # Might need this in the future:
    # if marks.counts.get(ParagraphAnnotation.ANY_ONE, 0) > 0:
    # score += ANY_ONE_WEIGHT
    max_score = marks.counts.get("max_score") or 0
    basic_score = score / max_score if max_score else 0.0
    return basic_score


def mark_question(question: str, model: str, day: date) -> ModelScores:
    """Compare a model's previously-extracted atoms against the marking scheme.

    Looks up atoms in the database for the given (question, model, day) combo
    and marks them according to the Paragraph Annotations defined by the marking
    scheme. Assumes that atoms have already been extracted AND that any repeats
    have been found (and labelled) for this response.
    """
    prompt = question
    scores = ModelScores()
    marking_scheme = service.create_or_get_marking_scheme(prompt)
    fact_to_para_id, paras_by_id = _build_fact_to_marking_para(prompt, marking_scheme)

    # Credit is per paragraph, not per point. Paragraph membership is 0/1.
    # Paragraphs with no points are left out: nothing can hit them.
    scoring_paras = {
        para_id: para for para_id, para in paras_by_id.items() if para.points
    }
    # Total positive weight available, so we can rescale scores to the range 0..1:
    para_weights = [
        ANNOTATION_WEIGHTS.get(para.annotation, 0.0) for para in scoring_paras.values()
    ]
    scores.counts["max_score"] = sum(w for w in para_weights if w > 0)

    response_atoms = service.get_points_for_range(prompt, model, day, day)
    if not response_atoms:
        # No atoms so nothing to mark!
        # TODO: log this as it might be a mistake;
        # need to extract atoms + find matches before marking.
        return scores

    n_essential_paras = sum(
        1
        for para in scoring_paras.values()
        if para.annotation is ParagraphAnnotation.ESSENTIAL
    )

    matched_para_ids: set[int] = set()
    for response_atom in response_atoms:
        # look for known matches or new "fuzzy" matches to old atoms:
        # check if response_atom is annotated in fact_to_para_id
        # else check if it is a repeat_of an atom that *has* been annotated
        para_id = fact_to_para_id.get(response_atom.atomic_fact)
        if para_id is None and response_atom.repeat_of:
            parent = service.get_point(response_atom.repeat_of)
            if parent:
                para_id = fact_to_para_id.get(parent.atomic_fact)

        if para_id is None:
            # this atom hasn't been annotated in the marking scheme
            scores.counts["unlabelled"] += 1
        elif para_id not in matched_para_ids:
            # we found this atom in the marking scheme; a paragraph earns its
            # mark once, however many of its points the response matches
            matched_para_ids.add(para_id)
            scores.counts[paras_by_id[para_id].annotation.value] += 1

    scores.counts["missed_essential"] = (
        n_essential_paras - scores.counts[ParagraphAnnotation.ESSENTIAL]
    )

    scores.counts["basic_score"] = calculate_question_score(scores)

    return scores


def civic_score(
    day_to_process: date,
    factual_scores: dict[str, dict[str, ModelScores]],
) -> dict[str, CivicsScore]:
    """Measure accuracy for questions tagged "civics" vs accuracy for other
    questions.

    Pass factual_scores from a fresh factuality pass."""
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()

    civics_score: dict[str, CivicsScore] = {}
    for model in all_models:
        counter = {
            "civics_total": 0.0,
            "civics_count": 0,
            "non_civics_total": 0.0,
            "non_civics_count": 0,
        }
        for question in all_questions:
            this_scores = factual_scores[question["question"]].get(model)
            if this_scores:
                if "civics" in question["tags"]:
                    counter["civics_count"] += 1
                    counter["civics_total"] += this_scores.counts["basic_score"]
                else:
                    counter["non_civics_count"] += 1
                    counter["non_civics_total"] += this_scores.counts["basic_score"]
            else:
                # TODO: make this a log message instead
                print(
                    f"No scores for {model} {question['question']}[0:25]... {day_to_process}"
                )
        civics_mean = (
            counter["civics_total"] / counter["civics_count"]
            if counter["civics_count"]
            else 0.0
        )
        non_civics_mean = (
            counter["non_civics_total"] / counter["non_civics_count"]
            if counter["non_civics_count"]
            else 0.0
        )
        civics_score[model] = {
            "civics_mean": civics_mean,
            "non_civics_mean": non_civics_mean,
            "overall_score": civics_mean,  # / non_civics_mean if non_civics_mean else 0.0,
            # for now, just use the mean civics score, ignoring non-civics scores
        }
    return civics_score


def transparency_score(
    day_to_process: date,
    factual_scores: dict[str, dict[str, ModelScores]],
) -> dict[str, TransparencyScore]:
    """Mean factual score for questions tagged 'policy' vs all others.

    Pass factual_scores from a fresh factuality pass."""
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()

    scores: dict[str, TransparencyScore] = {}
    for model in all_models:
        counter = {
            "balanced_total": 0.0,
            "balanced_count": 0,
            "other_total": 0.0,
            "other_count": 0,
        }
        for question in all_questions:
            this_scores = factual_scores[question["question"]].get(model)
            if this_scores:
                if "policy" in question["tags"]:
                    counter["balanced_count"] += 1
                    counter["balanced_total"] += this_scores.counts["basic_score"]
                else:
                    counter["other_count"] += 1
                    counter["other_total"] += this_scores.counts["basic_score"]
            else:
                print(
                    f"No scores for {model} {question['question'][:25]}... {day_to_process}"
                )
        balanced_mean = (
            counter["balanced_total"] / counter["balanced_count"]
            if counter["balanced_count"]
            else 0.0
        )
        other_mean = (
            counter["other_total"] / counter["other_count"]
            if counter["other_count"]
            else 0.0
        )
        scores[model] = {
            "transparency_mean": balanced_mean,
            "non_transparency_mean": other_mean,
            "overall_score": balanced_mean,
        }
    return scores


def timeliness_score(
    day_to_process: date, factual_scores: dict[str, dict[str, ModelScores]]
) -> dict[str, float]:
    """Score responses for how up-to-date their answers are.
    Only considers paragraphs with an 'applies_from' and/or an 'applies_to' field
    and penalises matches to out-of-date-range paragraphs.

    Pass factual_scores from a fresh factuality pass."""
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()

    timely_score_total = {model: 0.0 for model in all_models}
    timely_para_count = {model: 0 for model in all_models}
    timely_scores = dict()

    for question in all_questions:
        prompt = question["question"]

        marking_scheme = service.create_or_get_marking_scheme(prompt)
        # para_ids = service.get_all_paragraphs(question["question"])

        fact_to_para_id, paras_by_id = _build_fact_to_marking_para(
            prompt, marking_scheme
        )

        for model in all_models:
            response_atoms = service.get_points_for_range(
                prompt, model, day_to_process, day_to_process
            )
            for response_atom in response_atoms:
                # we've annotated this specific point, or its repeat_of parent
                para_id = fact_to_para_id.get(response_atom.atomic_fact)
                if para_id is None and response_atom.repeat_of:
                    parent = service.get_point(response_atom.repeat_of)
                    if parent:
                        para_id = fact_to_para_id.get(parent.atomic_fact)
                para = paras_by_id.get(para_id) if para_id is not None else None

                if para and (para.applies_from or para.applies_to):
                    response_date = response_atom.response_time.date()
                    in_range = (
                        para.applies_from is None or response_date >= para.applies_from
                    ) and (para.applies_to is None or response_date <= para.applies_to)
                    raw_marks = factual_scores[prompt].get(model)
                    if raw_marks:
                        timely_score_total[model] += calculate_question_score(
                            raw_marks, not in_range
                        )

                    timely_para_count[model] += 1

            timely_scores[model] = (
                timely_score_total[model] / timely_para_count[model]
                if timely_para_count[model]
                else 0.0
            )
    return timely_scores


def factual_score_all_responses(
    day_to_process: date,
) -> dict[str, dict[str, ModelScores]]:
    """Score all model responses for all questions on the given day.

    Returns a nested dict: {question_prompt: {model: ModelScores}}
    """
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()
    results: dict[str, dict[str, ModelScores]] = {}
    for question in all_questions:
        prompt = question["question"]
        results[prompt] = {}
        for model in all_models:
            try:
                results[prompt][model] = mark_question(prompt, model, day_to_process)
                print(".", end="")  # progress bar
            except Exception as e:
                print(f"Error scoring {model} / {prompt}: {e}")
    return results


def extract_atoms_one_day(day_to_process: date) -> None:
    """Use LLM to extract atomic facts and URLs from all model responses on the
    given day and write them to the database for later annotation and scoring.
    It doesn't do any duplication detection or 'repeat_of' labelling."""
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()
    logger.info(
        "extract_atoms_one_day(%s): %d question(s), %d model(s): %s",
        day_to_process,
        len(all_questions),
        len(all_models),
        all_models,
    )

    # Per-model outcome counts, logged as a summary at the end. A model stuck
    # at all-"no_response" (rather than a mix, or "error") points at a data
    # problem upstream (e.g. that day's generation hasn't landed, or was
    # served from the wrong llm-polygraph db) rather than a bug here.
    counts: dict[str, Counter[str]] = {model: Counter() for model in all_models}

    for id, question in enumerate(all_questions):
        logger.info("Q %d / %d on %s", id, len(all_questions) - 1, day_to_process)
        question_prompt = question["question"]
        for model in all_models:
            try:
                known_atoms = service.get_points_for_range(
                    question_prompt, model, day_to_process, day_to_process
                )
                if known_atoms:
                    logger.info(
                        "Already have %d atoms for %s/%s; skipping extraction",
                        len(known_atoms),
                        model,
                        day_to_process,
                    )
                    counts[model]["already_had"] += 1
                    continue

                response = llm_responses.get_response(
                    model, question_prompt, day_to_process
                )
                if not response:
                    active_db = llm_responses.get_active_db()
                    logger.warning(
                        "No response found for %s / %r on %s (active db: %r) "
                        "- skipping extraction",
                        model,
                        question_prompt,
                        day_to_process,
                        active_db,
                    )
                    counts[model]["no_response"] += 1
                    continue

                extracted = extract_atomic_facts(response)
                logger.info(
                    "Extracted %d atoms for %s / %r",
                    len(extracted),
                    model,
                    question_prompt,
                )
                service.add_points(extracted)
                counts[model]["extracted"] += 1

            except Exception:
                logger.exception(
                    "Failed to extract atoms for %s / %r on %s",
                    model,
                    question_prompt,
                    day_to_process,
                )
                counts[model]["error"] += 1

    for model, model_counts in counts.items():
        logger.info(
            "extract_atoms_one_day(%s) summary for %s: %s",
            day_to_process,
            model,
            dict(model_counts),
        )
