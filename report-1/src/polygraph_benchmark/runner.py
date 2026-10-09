# Top-level functions to process each day's data of the benchmark's LLM cron job
# flake8: noqa: E501

from dataclasses import asdict, dataclass
from datetime import date, timedelta

from api.models import AGGREGATE_QUESTION, CivicsScore, DailyScore, TransparencyScore
from api.service import MarkingSchemeService, PolygraphResponseService
from polygraph_benchmark import auto_marker, consistency, source_analysis
from polygraph_benchmark.auto_marker import (
    ModelScores,
    civic_score,
    extract_atoms_one_day,
    factual_score_all_responses,
    timeliness_score,
    transparency_score,
)
from polygraph_benchmark.consistency import (
    DayCounts,
    consistency_score_all_responses,
    find_repeated_atoms,
)
from polygraph_benchmark.source_analysis import extract_urls_one_day


@dataclass
class ScoresAccumulator:
    """Used to accumulate scores for all models and all questions while
    calculating metrics"""

    factual: dict[str, dict[str, ModelScores]]  # question → model → metric
    consistency: dict[str, dict[str, DayCounts]]  # question → model → metric
    civics: dict[str, CivicsScore]  # model -> metric
    timeliness: dict[str, float]  # model -> metric
    transparency: dict[str, TransparencyScore]  # model -> metric


def calc_averages(
    fact_scores: dict,
    consistency_scores: dict,
    all_questions: list[str],
    model: str,
) -> tuple[dict, dict]:
    """Calculate mean scores across all questions for one model
    and one day."""
    factual_dicts = [
        fact_scores[q][model].counts
        for q in all_questions
        if model in fact_scores.get(q, {})
    ]
    consistency_dicts = [
        asdict(consistency_scores[q][model])
        for q in all_questions
        if model in consistency_scores.get(q, {})
    ]
    mean_factual = (
        {
            k: sum(d[k] for d in factual_dicts) / len(factual_dicts)
            for k in factual_dicts[0]
        }
        if factual_dicts
        else {}
    )
    mean_consistency = (
        {
            k: sum(d[k] for d in consistency_dicts) / len(consistency_dicts)
            for k in consistency_dicts[0]
        }
        if consistency_dicts
        else {}
    )
    return mean_factual, mean_consistency


def calculate_and_update_scores(day_to_process: date) -> None:
    """Calculate each of the metrics and store them in the database"""

    # Calculate the factuality and consistency scores for each
    # question/model combination for this day:
    fact_scores = factual_score_all_responses(day_to_process)

    consistency_scores = consistency_score_all_responses(day_to_process)

    # All three derive from fact_scores, so hand it over rather than let them
    # re-read the previous run's stored scores.
    # Need to store civic_scores on a model/day basis (along with any other aggregate scores)
    # We use a dummy sentinal value "aggregate" in the question field
    civic_scores = civic_score(day_to_process, fact_scores)

    # most questions will never have time ranges, so we only calculate the aggregate scores
    timeliness_scores = timeliness_score(day_to_process, fact_scores)

    transparency_scores = transparency_score(day_to_process, fact_scores)

    day_scores = ScoresAccumulator(
        factual=fact_scores,
        consistency=consistency_scores,
        civics=civic_scores,
        timeliness=timeliness_scores,
        transparency=transparency_scores,
    )
    llm_responses = PolygraphResponseService()
    all_questions = [q["question"] for q in llm_responses.get_all_questions()]
    all_models = llm_responses.get_all_models()

    save_scores(day_to_process, day_scores, all_questions, all_models)


def save_scores(
    day_to_process: date,
    day_scores: ScoresAccumulator,
    all_questions: list[str],
    all_models: list[str],
) -> None:
    """Write a DayScores to the database: one aggregate row per model plus one
    row per question/model combination."""
    service = MarkingSchemeService()
    try:
        for model in all_models:
            # First, create the aggregate row, averaging across all questions:
            mean_factual, mean_consistency = calc_averages(
                day_scores.factual, day_scores.consistency, all_questions, model
            )
            service.store_daily_score(
                DailyScore(
                    response_date=day_to_process.isoformat(),
                    question=AGGREGATE_QUESTION,
                    model=model,
                    factual=mean_factual,
                    consistency=mean_consistency,
                    civics=day_scores.civics.get(model, CivicsScore()),
                    timeliness=day_scores.timeliness.get(model, 0.0),
                    transparency=day_scores.transparency.get(
                        model, TransparencyScore()
                    ),
                )
            )
            # Second, loop over all questions to store question-specific scores:
            for q in all_questions:
                fs = day_scores.factual.get(q, {}).get(model)
                cs = day_scores.consistency.get(q, {}).get(model)

                if fs is None or cs is None:
                    continue
                service.store_daily_score(
                    DailyScore(
                        response_date=day_to_process.isoformat(),
                        question=q,
                        model=model,
                        factual=fs.counts,
                        consistency=asdict(cs),
                        civics={},
                        timeliness=0.0,
                        transparency={},
                    )
                )
    finally:
        service.close()


def release_db_connections() -> None:
    """Close the database connections held by each stage's module-level service.

    Each stage of the daily run spends long stretches doing LLM calls or
    similarity scoring without touching the database, and idle connections get
    dropped underneath us. Handing them back between stages keeps them from
    idling across the next slow phase; they reopen on next use.
    """
    for module in (auto_marker, consistency, source_analysis):
        module.service.close()


def run_for_day(day_to_process: date):
    """Top level process to carry out full analysis of one day's LLM responses.
    First, atomic facts are extracted from each response and stored.
    Second, these atoms are checked to see if they repeat earlier atoms.
    Finally, the current marking scheme is applied and the scores written to the database.
    If 'just_mark' is set to True, then the first two (very slow!) steps are skipped, and
    the marking scheme is (re-)applied. This can be triggered by users when they edit the
    marking schemes.
    """

    # Use an LLM to extract atomic facts from all the responses for this day
    extract_atoms_one_day(day_to_process)
    release_db_connections()

    # Analyse the day's extracted atoms to see if any are (semantically)
    # repeats of earlier responses by the same model
    find_repeated_atoms(day_to_process)
    release_db_connections()

    # Find and store any URLs from each response
    extract_urls_one_day(day_to_process)
    release_db_connections()

    calculate_and_update_scores(day_to_process)
