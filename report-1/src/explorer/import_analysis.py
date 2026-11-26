"""Import the direct-reply analysis from a CSV into the response_analysis table.

The table is dropped and recreated on every run, so it always reflects the
current state of the external analysis file. Each row is matched to a response
by (question_id, model, date), which responses declares UNIQUE.

The response's opening text is stored alongside the labels, because the external
script normalises its whitespace and cuts it at the first full sentence, so it
can't be reproduced from responses.response_text.

Every other path opens the export read-only (see api.service), so this script
opens its own writable connection. It only ever touches response_analysis;
nothing else creates or reads that table. Stop the explorer first if it is
running against the same file, or the write may hit "database is locked".

Usage:
    uv run python src/explorer/import_analysis.py
    uv run python src/explorer/import_analysis.py --csv data/direct_replies.csv
"""

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

from api.service import DEFAULT_DB_PATH

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "s1_direct_replies.csv"
)

# the opening of the response, as the external script cut and normalised it
TEXT_COLUMNS = ["start"]
# the analysis labels, stored verbatim; the label -> score mapping stays in
# factuality_viewer.DIRECT_REPLY_VALUES so there's one source of truth for it
LABEL_COLUMNS = ["polarity", "certainty", "precision", "directness"]
# scope flags: each cell holds either the column's own name or an empty string
FLAG_COLUMNS = ["temporal", "geographic", "definitional", "attributional"]
# the insert order, matching SCHEMA below
VALUE_COLUMNS = [*TEXT_COLUMNS, *LABEL_COLUMNS, *FLAG_COLUMNS]

SCHEMA = """
CREATE TABLE response_analysis (
    response_id INTEGER PRIMARY KEY REFERENCES responses (id),
    start TEXT NOT NULL,           -- the response's opening, as analysed
    polarity TEXT NOT NULL,        -- positive, negative, neutral, both
    certainty TEXT NOT NULL,       -- high, low
    precision TEXT NOT NULL,       -- high, low
    directness TEXT NOT NULL,      -- direct, indirect
    temporal INTEGER NOT NULL,     -- 1 if the claim is scoped in time
    geographic INTEGER NOT NULL,   -- 1 if scoped to a place
    definitional INTEGER NOT NULL, -- 1 if scoped by a definition
    attributional INTEGER NOT NULL -- 1 if attributed to a source
)
"""


def read_csv_rows(path: Path) -> list[dict]:
    # return the CSV's rows as dicts; the stdlib reader keeps the blank scope
    # flags as "", where pandas would turn them into NaN floats
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_response_ids(conn: sqlite3.Connection) -> dict[tuple, int]:
    # return the id of every response by (question_id, model, response_date),
    # which is unique, so the whole lookup can be done in memory
    rows = conn.execute("SELECT id, question_id, model, response_date FROM responses")
    return {(r[1], r[2], r[3]): r[0] for r in rows}


def match_rows(
    rows: list[dict], response_ids: dict[tuple, int]
) -> tuple[list[tuple], list[dict]]:
    # return (values to insert, rows with no matching response); question_id is
    # cast to int because responses.question_id is INTEGER and the CSV gives "2"
    values, unmatched = [], []
    for row in rows:
        key = (int(row["question_id"]), row["model"], row["date"])
        response_id = response_ids.get(key)
        if response_id is None:
            unmatched.append(row)
            continue
        values.append(
            (
                response_id,
                *(row[column] for column in TEXT_COLUMNS),
                *(row[label] for label in LABEL_COLUMNS),
                *(int(bool(row[flag])) for flag in FLAG_COLUMNS),
            )
        )
    return values, unmatched


def write_table(conn: sqlite3.Connection, values: list[tuple]) -> None:
    # drop and recreate the table, then insert, all in one transaction so a
    # failure can't leave a half-built table behind
    columns = ["response_id", *VALUE_COLUMNS]
    placeholders = ", ".join("?" * len(columns))
    with conn:
        conn.execute("DROP TABLE IF EXISTS response_analysis")
        conn.execute(SCHEMA)
        conn.executemany(
            f"INSERT INTO response_analysis ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )


def describe_unmatched(conn: sqlite3.Connection, unmatched: list[dict]) -> None:
    # say why rows found no response: an unknown question, a date outside the
    # export's range, an unknown model, or none of those
    questions = {r[0] for r in conn.execute("SELECT id FROM questions")}
    models = {r[0] for r in conn.execute("SELECT DISTINCT model FROM responses")}
    first, last = conn.execute(
        "SELECT MIN(response_date), MAX(response_date) FROM responses"
    ).fetchone()
    reasons: Counter[str] = Counter()
    for row in unmatched:
        why = []
        if int(row["question_id"]) not in questions:
            why.append("unknown question")
        if not first <= row["date"] <= last:
            why.append("date out of range")
        if row["model"] not in models:
            why.append("unknown model")
        reasons[", ".join(why) or "no response that day"] += 1
    print(f"\nWARNING: {len(unmatched):,} rows matched no response:")
    for reason, count in reasons.most_common():
        print(f"  {count:,} {reason}")
    print(f"  the database covers {first} to {last}")


def report(values: list[tuple]) -> None:
    # count the values of each label and flag that was imported; the values are
    # in VALUE_COLUMNS order after response_id, so column i + 1 holds its i'th
    counts = {
        column: Counter(value[i + 1] for value in values)
        for i, column in enumerate(VALUE_COLUMNS)
        if column not in TEXT_COLUMNS
    }
    for label in LABEL_COLUMNS:
        labels = "   ".join(f"{v} {n:,}" for v, n in counts[label].most_common())
        print(f"{label:<12}  {labels}")
    print("   ".join(f"{flag} {counts[flag][1]:,}" for flag in FLAG_COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))
    args = parser.parse_args()

    rows = read_csv_rows(args.csv)
    print(f"Read {len(rows):,} rows from {args.csv.name}")

    conn = sqlite3.connect(args.db)
    try:
        values, unmatched = match_rows(rows, load_response_ids(conn))
        write_table(conn, values)
        print(f"Matched {len(values):,} of {len(rows):,} responses")
        print(f"Inserted {len(values):,} rows into response_analysis")
        if unmatched:
            describe_unmatched(conn, unmatched)
        if values:
            print()
            report(values)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
