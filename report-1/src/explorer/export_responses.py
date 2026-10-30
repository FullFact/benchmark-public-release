"""Export LLM responses as JSONL, one response per line, for the external
direct-replies script (which produces data/direct_replies.csv).

Each line looks like:
{"question_id": 2, "question": "...", "model": "...", "provider": "...",
 "date": "2026-10-14", "response": "..."}

This can then be processed by
data-science-scripts/benchmark_response_analysis/direct_replies.py to estimate the labels.

Usage:
    uv run python src/explorer/export_responses.py --start 2026-09-01 --end 2026-09-22
"""

import argparse
import datetime
import json
from pathlib import Path

from api.service import PolygraphResponseService

S1_START = datetime.date(2026, 7, 13)
S1_END = datetime.date(2026, 9, 22)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "responses.jsonl"


def get_response_rows(
    start: datetime.date, end: datetime.date, models: list[str] | None = None
) -> list[dict]:
    # return one row per (question, model, date) that has a response text
    service = PolygraphResponseService()
    models = models or service.get_all_models()
    rows = []
    for question in service.get_all_questions():
        for model in models:
            day = start
            while day <= end:
                response = service.get_response(model, question["question"], day)
                if response:
                    rows.append(
                        {
                            "question_id": question["id"],
                            "question": question["question"],
                            "model": model,
                            "provider": response["provider"],
                            "date": day.isoformat(),
                            "response": response["response"],
                        }
                    )
                day += datetime.timedelta(days=1)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=datetime.date.fromisoformat, default=S1_START)
    parser.add_argument("--end", type=datetime.date.fromisoformat, default=S1_END)
    parser.add_argument("--models", nargs="+", help="default: all models")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rows = get_response_rows(args.start, args.end, args.models)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} responses to {args.output}")


if __name__ == "__main__":
    main()
