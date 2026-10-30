"""Daily mean scores over time, per model and dimension, for the time-series
page of the explorer app."""

import pandas as pd

from explorer.factuality_viewer import filter_by_dimension, filter_by_model

# model name meaning "every model combined"
ALL_MODELS = "all"


def date_range(df: pd.DataFrame) -> list[str]:
    # every date from the first to the last response date, with no gaps
    dates = pd.to_datetime(df["response_date"])
    return [
        d.date().isoformat() for d in pd.date_range(dates.min(), dates.max(), freq="D")
    ]


def daily_mean_scores(
    df: pd.DataFrame, dimension: str, model: str = ALL_MODELS
) -> pd.Series:
    # mean score per response date for one dimension and model (or all models);
    # NaN scores are skipped
    subset = filter_by_dimension(dimension, df)
    if model != ALL_MODELS:
        subset = filter_by_model(model, subset)
    return subset.groupby("response_date")["score"].mean()


def build_timeseries(
    df: pd.DataFrame,
    models: list[str],
    dimensions: list[str],
    question_ids: list[int] | None = None,
) -> dict:
    # one series per (model, dimension) pair, aligned to a shared date axis,
    # optionally restricted to some questions; days with no score are None so
    # the chart shows a gap
    dates = date_range(df)
    if question_ids is not None:
        df = df[df["question_id"].isin(question_ids)]
    series = []
    for model in models:
        for dimension in dimensions:
            means = daily_mean_scores(df, dimension, model)
            series.append(
                {
                    "model": model,
                    "dimension": dimension,
                    "values": [
                        None if pd.isna(means.get(d)) else float(means[d])
                        for d in dates
                    ],
                }
            )
    return {"dates": dates, "series": series}
