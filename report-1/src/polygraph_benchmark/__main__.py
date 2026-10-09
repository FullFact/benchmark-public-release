# Top-level functions to process each day's data of the benchmark's LLM cron job
# flake8: noqa: E501

import json
from datetime import datetime, timedelta

import click

from api.service import MarkingSchemeService
from polygraph_benchmark.runner import run_for_day


@click.command()
@click.option(
    "--start-date",
    default=lambda: datetime.now().strftime("%Y-%m-%d"),
    show_default="today",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First day to process, in YYYY-MM-DD format.",
)
@click.option(
    "--n-days",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of consecutive days to process, starting from --start-date.",
)
@click.option(
    "--output",
    default="scores.json",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help="File to write the calculated scores to.",
)
def cli(start_date: datetime, n_days: int, output: str) -> None:
    """Run the daily benchmark pipeline for a range of dates."""
    start = start_date.date()
    for offset in range(n_days):
        current = start + timedelta(days=offset)
        print(f"\n{'=' * 60}\nProcessing {current}\n{'=' * 60}")
        run_for_day(current)

    # In production, scores are written to the database as they're calculated.
    # Here, they're held in memory, so save them to a file at the end.
    scores = MarkingSchemeService().get_stored_scores()
    with open(output, "w") as f:
        json.dump([s.model_dump() for s in scores], f, indent=2)
    print(f"Wrote {len(scores)} scores to {output}")


if __name__ == "__main__":
    cli()
