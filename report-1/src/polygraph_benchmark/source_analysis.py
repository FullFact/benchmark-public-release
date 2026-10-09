# Analyse URLs returned by models

import re
from datetime import date
from urllib.parse import urlparse

from api.service import MarkingSchemeService, PolygraphResponseService

llm_responses = PolygraphResponseService()
service = MarkingSchemeService()


def get_urls_from_response(
    question_prompt: str, model: str, response_time: date
) -> dict[str, list[str]]:
    """Extract URLs from a model response and store in the db.
    Return ``{netloc: [full_urls]}``.

    Results are cached in the database — subsequent calls for the same
    (question, model, date) return the stored data without re-fetching."""
    cached = service.get_response_urls(question_prompt, model, response_time)
    if cached:
        return cached

    response = llm_responses.get_response(model, question_prompt, response_time)
    if not response:
        return dict()
    response_text = response["response"]

    raw_urls = re.findall(r"https?://[^\s)\]>\"']+", response_text)

    netloc_urls: dict[str, list[str]] = {}
    for url in raw_urls:
        netloc = urlparse(url).netloc
        if netloc:
            netloc_urls.setdefault(netloc, [])
            if url not in netloc_urls[netloc]:
                netloc_urls[netloc].append(url)

    service.store_response_urls(question_prompt, model, response_time, netloc_urls)
    return netloc_urls


def extract_urls_one_day(day_to_process: date) -> None:
    """Parse through all responses for one day and store any source URLs in the db"""
    all_questions = llm_responses.get_all_questions()
    all_models = llm_responses.get_all_models()
    for question in all_questions:
        question_prompt = question["question"]
        for model in all_models:
            get_urls_from_response(question_prompt, model, day_to_process)
