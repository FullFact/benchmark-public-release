import datetime
from pathlib import Path

import pandas as pd

from api.service import MarkingSchemeService, PolygraphResponseService

S1_START = datetime.date(2026, 7, 13)
S1_END = datetime.date(2026, 9, 22)

DIRECT_REPLIES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "s1_direct_replies.csv"
)
# direct-reply labels, used as extra score dimensions, with the numeric score
# for each label value
DIRECT_REPLY_VALUES = {
    # "neutral" and "both" have no polarity score; they're flagged by the
    # non_polar and bipolar dimensions instead
    "polarity": {"positive": 1, "negative": -1},
    "non_polar": {"positive": 0, "negative": 0, "neutral": 1, "both": 0},
    "bipolar": {"positive": 0, "negative": 0, "neutral": 0, "both": 1},
    "certainty": {"low": 0, "high": 1},
    "precision": {"low": 0, "high": 1},
    "directness": {"direct": 1, "indirect": 0},
}

# numeric score for each URL rating in url_annotations; a response's url_quality
# is the mean over its rated URLs
URL_RATING_VALUES = {"primary": 3, "good": 2, "OK": 1, "poor": -1}
URL_QUALITY = "url_quality"


# display order for dimensions; any others follow alphabetically
DIMENSION_ORDER = [
    "factuality",
    "certainty",
    "precision",
    "directness",
    "polarity",
    "bipolar",
    "non_polar",
    "consistency",
    URL_QUALITY,
]

YES_NO_QUESTIONS = [2, 15, 18, 48, 54, 84, 101]
# extra tag given to YES_NO_QUESTIONS, as if it were in the question-tag table
YES_NO_TAG = "yes/no"


def order_dimensions(dimensions: set[str]) -> list[str]:
    # return the dimensions in DIMENSION_ORDER, then any others alphabetically
    known = [d for d in DIMENSION_ORDER if d in dimensions]
    return known + sorted(dimensions - set(DIMENSION_ORDER))


def load_direct_replies(path: Path = DIRECT_REPLIES_PATH) -> pd.DataFrame:
    # return one row per (question, model, date, direct-reply label) in the
    # same shape as the daily scores, with the label as the dimension, plus the
    # start of the response
    df = pd.read_csv(path)
    df["non_polar"] = df["polarity"]
    df["bipolar"] = df["polarity"]
    for label, values in DIRECT_REPLY_VALUES.items():
        df[label] = df[label].map(values)
    return df.rename(columns={"date": "response_date"}).melt(
        id_vars=["response_date", "question_id", "model", "question", "start"],
        value_vars=list(DIRECT_REPLY_VALUES),
        var_name="dimension",
        value_name="score",
    )


def load_url_details(
    start: datetime.date = S1_START, end: datetime.date = S1_END
) -> pd.DataFrame:
    # return one row per rated URL cited in a response, in cited order
    return pd.DataFrame(MarkingSchemeService().get_url_ratings(start, end))


def load_url_ratings(
    start: datetime.date = S1_START, end: datetime.date = S1_END
) -> pd.DataFrame:
    # return one url_quality row per response with at least one rated URL, in
    # the same shape as the daily scores: the mean of its URL ratings
    df = load_url_details(start, end)
    df["score"] = df["rating"].map(URL_RATING_VALUES)
    df = (
        df.groupby(["response_date", "question_id", "model", "question"])["score"]
        .mean()
        .reset_index()
    )
    df["dimension"] = URL_QUALITY
    return df


def response_rows(
    df: pd.DataFrame,
    url_ratings: pd.DataFrame,
    model: str,
    question_id: int,
    dimension: str,
) -> list[dict]:
    # return one row per day on which the model answered the question, in date
    # order, with that day's score for the dimension (None if missing), the
    # start of the response and, for url_quality, the rated URLs it cited
    df = df[(df["model"] == model) & (df["question_id"] == question_id)]
    days = df.groupby("response_date")["start"].first()
    scores = df[df["dimension"] == dimension].set_index("response_date")["score"]
    urls: dict[str, list[dict]] = {}
    if dimension == URL_QUALITY:
        cited = url_ratings[
            (url_ratings["model"] == model) & (url_ratings["question_id"] == question_id)
        ]
        for day, group in cited.groupby("response_date"):
            urls[day] = group[["url", "rating"]].to_dict("records")
    return [
        {
            "date": day,
            "score": None if pd.isna(scores.get(day)) else float(scores[day]),
            "start": None if pd.isna(start) else start,
            "urls": urls.get(day, []),
        }
        for day, start in days.items()
    ]


def filter_by_model(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    # return the rows with model == model_name
    return df[df["model"] == model_name]


def load_questions() -> dict[int, dict]:
    # return every question by id, as {"id", "question", "tags"}, with the
    # yes/no tag added to YES_NO_QUESTIONS
    questions = {q["id"]: q for q in PolygraphResponseService().get_all_questions()}
    for question_id in YES_NO_QUESTIONS:
        if question_id in questions:
            questions[question_id]["tags"].append(YES_NO_TAG)
    return questions


def filter_by_tag(
    tag: str, df: pd.DataFrame, questions: dict[int, dict]
) -> pd.DataFrame:
    # return the rows for questions with the given tag
    tagged = [q_id for q_id, q in questions.items() if tag in q["tags"]]
    return df[df["question_id"].isin(tagged)]


def filter_by_dimension(dim_name: str, df: pd.DataFrame) -> pd.DataFrame:
    # return the rows with dimension == dim_name
    return df[df["dimension"] == dim_name]


def summarise_score(df: pd.DataFrame) -> pd.Series:
    # calculate and return min, max, mean (NULL scores are skipped) and row count
    summary = df["score"].agg(["min", "max", "mean", "size"])
    summary["nan_count"] = df["score"].isna().sum()
    return summary


def summarise_dimensions(df: pd.DataFrame) -> list[dict]:
    # return a summary per dimension, skipping dimensions that don't apply
    # (every score is NaN); NaN becomes None so the result serialises to JSON
    summaries = []
    for dimension in order_dimensions(set(df["dimension"])):
        scores = summarise_score(filter_by_dimension(dimension, df))
        if scores["nan_count"] == scores["size"]:
            continue
        summaries.append(
            {
                "dimension": dimension,
                "mean": None if pd.isna(scores["mean"]) else float(scores["mean"]),
                "min": None if pd.isna(scores["min"]) else float(scores["min"]),
                "max": None if pd.isna(scores["max"]) else float(scores["max"]),
                "size": int(scores["size"]),
                "nan_count": int(scores["nan_count"]),
            }
        )
    return summaries


def load_all_scores(
    start: datetime.date = S1_START, end: datetime.date = S1_END
) -> pd.DataFrame:
    # return the daily scores with the direct-reply labels and URL quality added
    # as extra dimensions, and every row given the start of its response (NaN where
    # there's no direct-reply data)
    df_all_scores = pd.DataFrame(MarkingSchemeService().get_daily_scores(start, end))
    df_replies = load_direct_replies()
    keys = ["question_id", "model", "response_date"]
    starts = df_replies[keys + ["start"]].drop_duplicates(keys)
    df_joined = pd.concat(
        [
            df_all_scores,
            df_replies.drop(columns="start"),
            load_url_ratings(start, end),
        ],
        ignore_index=True,
    )
    return df_joined.merge(starts, on=keys, how="left")


def display_summary(scores: pd.Series, label: str | None = None) -> None:
    # skip dimensions that don't apply (every score is NaN)
    if scores["nan_count"] == scores["size"]:
        return
    if label:
        print(f"{label:20s}", end="\t")
    print(
        f"Mean: {scores['mean']:.4f}   ({scores['min']:.2f} - {scores['max']:.2f})  ({int(scores['size'])} rows, Missing: {int(scores['nan_count'])})"
    )


def representative_answers(df: pd.DataFrame, dim_name: str, n: int = 5) -> pd.DataFrame:
    # pick n rows for dim_name whose scores are spread evenly from the lowest
    # to the highest score (e.g. 0, 0.25, 0.5, 0.75, 1), taking the closest
    # unused row to each target; rows with no score are skipped
    df_dim = filter_by_dimension(dim_name, df).dropna(subset="score")
    df_dim = df_dim.sort_values("score", kind="stable")
    if len(df_dim) > n:
        lo, hi = df_dim["score"].min(), df_dim["score"].max()
        targets = [lo + k * (hi - lo) / max(n - 1, 1) for k in range(n)]
        remaining = df_dim
        chosen = []
        for target in targets:
            row_id = (remaining["score"] - target).abs().idxmin()
            chosen.append(row_id)
            remaining = remaining.drop(row_id)
        df_dim = df_dim.loc[chosen]
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(df_dim)
    return df_dim


def summarise_scores_by_question(df: pd.DataFrame) -> None:
    # display each question's id, text and tags, then a summary per dimension
    questions = load_questions()
    dimensions = order_dimensions(set(df["dimension"]))
    for question_id in sorted(set(df["question_id"])):
        question = questions[question_id]
        print(f"\n[{question_id}] {question['question']}")
        print(f"Tags: {', '.join(question['tags']) or '-'}")
        df_question = df[df["question_id"] == question_id]
        for dimension in dimensions:
            subset = filter_by_dimension(dimension, df_question)
            display_summary(summarise_score(subset), f"  {dimension}")


def main():
    df_joined = load_all_scores()

    # subset = filter_by_dimension("factuality", df_joined)
    # display_summary(summarise_score(subset), "factuality")
    # subset = filter_by_dimension("civic_responsibility", df_joined)
    # display_summary(summarise_score(subset), "civic_responsibility")

    all_models = set(df_joined["model"])
    for model in all_models:
        subset = filter_by_dimension("factuality", filter_by_model(model, df_joined))
        display_summary(summarise_score(subset), f"factuality for {model:20s} ")

    # for label in DIRECT_REPLY_VALUES:
    #     subset = filter_by_dimension(label, df_joined)
    #     display_summary(summarise_score(subset), label)

    summarise_scores_by_question(df_joined)
    question_id = 15  # 142=who is PM?
    df_question = df_joined[df_joined["question_id"] == question_id]
    # subset = filter_by_dimension("factuality", df_question)
    subset = filter_by_model("claude-sonnet-5", df_question)
    reps = representative_answers(subset, "factuality")
    for index, row in reps.iterrows():
        print(f"{row['score']:.4f}  {row['start']}")


if __name__ == "__main__":
    main()
