"""Demo: list the benchmark questions and their tags.

Reads from the local SQLite file via the PolygraphResponseService.
Run from the report-1 folder with:

    uv run python scripts/demo.py
"""

from datetime import date

from api.service import PolygraphResponseService


def questions_report() -> None:
    llm_responses = PolygraphResponseService()
    questions = llm_responses.get_all_questions()

    print(f"{len(questions)} benchmark questions:\n")
    for i, question in enumerate(questions, start=1):
        tags = ", ".join(question["tags"]) or "no tags"
        print(f"{i:>3}. {question['question']}  \t[{tags}]")


def response_report(model: str, question: str, response_date: date) -> dict | None:
    llm_responses = PolygraphResponseService()
    response = llm_responses.get_response(model, question, response_date)

    print(f"{question}\nResponse by '{model}' on {response_date}:\n")
    if not response:
        print("No response found.")
        return None
    print(response["response"])
    return response


if __name__ == "__main__":
    questions_report()
    response_report("gpt-5", "When did women get the right to vote?", date(2026, 9, 1))
