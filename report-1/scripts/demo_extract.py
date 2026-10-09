from datetime import date

from api.service import PolygraphResponseService
from polygraph_benchmark.auto_marker import extract_atomic_facts


def extraction_demo(response: dict) -> None:
    """Extract atomic facts from a made-up response. Calls Gemini, so needs
    GEMINI_API_KEY to be set."""

    print(
        f"Extracting atoms from:\n{response['prompt']}\nMade-up response:\n\n{response['response']}\n"
    )
    atoms = extract_atomic_facts(response)
    print(f"Extracted {len(atoms)} atomic facts:")
    for atom in atoms:
        evidence = f"  ({', '.join(atom.evidence)})" if atom.evidence else ""
        print(f"  - {atom.atomic_fact}{evidence}")


def response_report(model: str, question: str, response_date: date) -> dict | None:
    llm_responses = PolygraphResponseService()
    response = llm_responses.get_response(model, question, response_date)
    return response


if __name__ == "__main__":
    response = response_report(
        "gpt-5", "When did women get the right to vote?", date(2026, 9, 1)
    )
    if response:
        extraction_demo(response)
    else:
        print("Error: failed to find response")
