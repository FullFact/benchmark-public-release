"""Demo: summarise the benchmark questions, atoms etc.

Reads from the local SQLite file via the PolygraphResponseService.
Run from the report-1 folder with:

    uv run python scripts/demo_data_reports.py
"""

from datetime import date

from api.service import PolygraphResponseService

MAX_ATOMS = 20  # higher atom counts are grouped to a "20 or more" row for simplicity
START_DATE = date(2026, 7, 15)  # reports only include data from this date onwards


def questions_report() -> None:
    """Just list all the questions"""
    llm_responses = PolygraphResponseService()
    questions = llm_responses.get_all_questions()

    print(f"{len(questions)} benchmark questions:\n")
    for i, question in enumerate(questions, start=1):
        tags = ", ".join(question["tags"]) or "no tags"
        print(f"{i:>3}. {question['question']}  \t[{tags}]")


def response_report(model: str, question: str, response_date: date) -> dict | None:
    """Display a single chatbot response."""
    llm_responses = PolygraphResponseService()
    response = llm_responses.get_response(model, question, response_date)

    print(f"{question}\nResponse by '{model}' on {response_date}:\n")
    if not response:
        print("No response found.")
        return None
    print(response["response"])
    return response


def atom_counts_report(start_date: date = START_DATE) -> dict[int, int]:
    """Count responses and atoms from start_date onwards, and show how many atoms
    each response produced.
    Returns {number of atoms: number of responses with that many atoms}, where
    the last key, MAX_ATOMS, counts responses with that many atoms or more."""
    conn = PolygraphResponseService().store.conn
    start = (start_date.isoformat(),)
    n_responses = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE response_date >= ?", start
    ).fetchone()[0]
    # An atom's fact_date is the date of the response it came from.
    n_atoms = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE fact_date >= ?", start
    ).fetchone()[0]
    n_response_atoms = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE response_id IS NOT NULL AND fact_date >= ?",
        start,
    ).fetchone()[0]
    # repeat_of links an atom to an earlier atom that says the same thing.
    n_repeats = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE repeat_of IS NOT NULL AND fact_date >= ?",
        start,
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT MIN(n_atoms, ?) AS n_atoms, COUNT(*) FROM ("
        "  SELECT r.id, COUNT(a.id) AS n_atoms FROM responses r"
        "  LEFT JOIN atoms a ON a.response_id = r.id"
        "  WHERE r.response_date >= ? GROUP BY r.id"
        ") GROUP BY 1 ORDER BY 1",
        (MAX_ATOMS, start_date.isoformat()),
    )
    atoms_per_response = {n_atoms: n for n_atoms, n in rows}

    print(f"From {start_date} onwards:")
    print(f"Responses: {n_responses:,}")
    print(f"Atoms: {n_atoms:,}")
    print(
        f"  of which {n_response_atoms:,} come from chatbot responses; the rest are "
        "staff-written"
    )
    print(f"  {n_repeats:,} ({n_repeats / n_atoms:.0%}) repeat an earlier atom")
    print(
        f"  {n_atoms - n_repeats:,} ({(n_atoms - n_repeats) / n_atoms:.0%}) "
        "are not repeats"
    )
    print(f"Mean atoms per response: {n_response_atoms / n_responses:.1f}\n")

    print("Atoms per response:")
    print(f"{'Atoms':>10} {'Responses':>10}")
    scale = max(atoms_per_response.values()) / 50
    for n_atoms, n in atoms_per_response.items():
        label = f"{n_atoms} or more" if n_atoms == MAX_ATOMS else str(n_atoms)
        print(f"{label:>10} {n:>10,}  {'#' * round(n / scale)}")
    return atoms_per_response


if __name__ == "__main__":
    questions_report()
    response_report("gpt-5", "When did women get the right to vote?", date(2026, 9, 1))
    atom_counts_report()
