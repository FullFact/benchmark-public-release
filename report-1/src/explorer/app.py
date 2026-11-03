"""Litestar API and web pages for exploring benchmark scores interactively:
an intro (/), the score explorer (/scores), time series (/timeseries) and
individual responses (/responses).

All data is loaded once at startup, so request handlers only work on the
in-memory DataFrame (and never touch the SQLite connection, which can't be
shared across threads).

Usage:
    uv run python src/explorer/app.py [--port 8000]
then open http://localhost:8000
"""

import argparse
from pathlib import Path

import pandas as pd
import uvicorn
from litestar import Litestar, MediaType, get
from litestar.datastructures import State
from litestar.exceptions import ValidationException

from explorer.factuality_viewer import (
    filter_by_model,
    filter_by_tag,
    load_all_scores,
    load_questions,
    load_url_details,
    FACTUALITY_BANDS,
    factuality_band_rows,
    order_dimensions,
    question_table,
    response_rows,
    summarise_dimensions,
)
from explorer.timeseries import ALL_MODELS, build_timeseries

STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_data(app: Litestar) -> None:
    app.state.df = load_all_scores()
    app.state.questions = load_questions()
    app.state.url_ratings = load_url_details()


@get("/", media_type=MediaType.HTML)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@get("/scores", media_type=MediaType.HTML)
async def scores_page() -> str:
    return (STATIC_DIR / "scores.html").read_text(encoding="utf-8")


@get("/questions", media_type=MediaType.HTML)
async def questions_page() -> str:
    return (STATIC_DIR / "questions.html").read_text(encoding="utf-8")


@get("/timeseries", media_type=MediaType.HTML)
async def timeseries_page() -> str:
    return (STATIC_DIR / "timeseries.html").read_text(encoding="utf-8")


@get("/responses", media_type=MediaType.HTML)
async def responses_page() -> str:
    return (STATIC_DIR / "responses.html").read_text(encoding="utf-8")


@get("/api/models")
async def get_models(state: State) -> list[str]:
    return sorted(set(state.df["model"]))


@get("/api/dimensions")
async def get_dimensions(state: State) -> list[str]:
    # every dimension in display order
    return order_dimensions(set(state.df["dimension"]))


@get("/api/questions")
async def get_questions(state: State, model: str | None = None) -> list[dict]:
    # questions that have scores (from the model, if given), in id order
    df: pd.DataFrame = state.df
    if model:
        df = filter_by_model(model, df)
    return [
        state.questions[question_id]
        for question_id in sorted(set(df["question_id"]))
    ]


@get("/api/tags")
async def get_tags(state: State) -> list[str]:
    # every tag used by a question that has scores
    question_ids = set(state.df["question_id"])
    return sorted(
        {
            tag
            for q_id, q in state.questions.items()
            if q_id in question_ids
            for tag in q["tags"]
        }
    )


@get("/api/summary")
async def get_summary(
    state: State,
    model: str | None = None,
    question_id: int | None = None,
    tag: str | None = None,
) -> dict:
    # per-dimension summary for one model, question and/or question tag
    # (omitted = all), plus the factuality correctness bands in band order
    df: pd.DataFrame = state.df
    if model:
        df = filter_by_model(model, df)
    if tag:
        df = filter_by_tag(tag, df, state.questions)
    if question_id is not None:
        df = df[df["question_id"] == question_id]
    return {
        "model": model,
        "question_id": question_id,
        "tag": tag,
        "dimensions": summarise_dimensions(df),
        "factuality_bands": sorted(
            summarise_dimensions(factuality_band_rows(df)),
            key=lambda d: FACTUALITY_BANDS.index(d["dimension"]),
        ),
    }


@get("/api/timeseries")
async def get_timeseries(
    state: State,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    tag: str | None = None,
    question_id: int | None = None,
) -> dict:
    # daily mean score per (model, dimension), optionally for one question tag
    # and/or question; "all" means every model combined
    question_ids = None
    if tag:
        question_ids = [q_id for q_id, q in state.questions.items() if tag in q["tags"]]
    if question_id is not None:
        question_ids = [
            q_id for q_id in question_ids or [question_id] if q_id == question_id
        ]
    return build_timeseries(
        state.df, models or [ALL_MODELS], dimensions or ["factuality"], question_ids
    )


@get("/api/responses")
async def get_responses(
    state: State, model: str, question_id: int, dimension: str
) -> dict:
    # one model's daily responses to one question, with their scores for one
    # dimension
    return {
        "rows": response_rows(
            state.df, state.url_ratings, model, question_id, dimension
        )
    }


@get("/api/question-table")
async def get_question_table(
    state: State, model: str | None = None, dimension: str | None = None
) -> dict:
    # mean score per question: one column per dimension for a model ("all" for
    # every model combined), or one column per model for a dimension
    if (model is None) == (dimension is None):
        raise ValidationException("Give exactly one of model or dimension")
    table = question_table(
        state.df, None if model == ALL_MODELS else model, dimension
    )
    for row in table["rows"]:
        question = state.questions[row["question_id"]]
        row["question"] = question["question"]
        row["tags"] = question["tags"]
    return table


app = Litestar(
    route_handlers=[
        index,
        scores_page,
        questions_page,
        timeseries_page,
        responses_page,
        get_models,
        get_dimensions,
        get_questions,
        get_tags,
        get_summary,
        get_timeseries,
        get_responses,
        get_question_table,
    ],
    on_startup=[load_data],
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark score explorer")
    parser.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, port=parser.parse_args().port)
