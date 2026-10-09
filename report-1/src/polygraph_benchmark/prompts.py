"""Prompts used to extract atomic facts from LLM responses."""

# flake8: noqa: E501

ATOMIC_FACTS_EXTRACTION_PROMPT = """Your task is to help analyse essay-style answers to questions.
    The person you are helping is an expert academic in this field.
    You will be given a question and a short essay. Please identify the key facts
    in the essay that are relevant to the question. Do not try and summarise
    the whole essay.

    First, select up to twenty of the most important statements.
    You should ignore background information, minor details and statements that
    are clearly irrelevant to the question. If there are less than 20 important
    statements, then only select those.

    Second, break these statements down to the simplest form expressing a single
    "atomic" fact, without any subordinate clauses or background
    context. Each sentence you are given will probably contain 3-5 facts. If there
    are any URLs provided to statement, be sure to keep the URLs with the atomic
    fact you extract.

    Third, rephrase each fact as necessary to make it self-contained and easy
    to understand. Keep any URLs unchanged, exactly as originally provided.

    Finally, select up to 20 distinct facts that are most relevant to the question.

    Do not add to or correct any information there, but just summarise the key facts.
    Return the facts as a structured list. For each fact, provide the atomic_fact
    text and the url if one was cited alongside it (otherwise null).

    Here is an example.

    The question is ```Who wrote 'A Tale of Two Cities' and when?``` and
    the provided answer is:

    A Tale of Two Cities was written by Charles Dickens in 1859 [https://en.wikipedia.org/wiki/Charles_Dickens].
    It is set in London and Paris before and during the French Revolution.
    The opening line is "It was the best of times, it was the worst of times."
    Dickens was 47 years old when he wrote it.

    Then your response should be a list like:
    [
      {{"atomic_fact": "A Tale of Two Cities was written by Charles Dickens", "url": "https://en.wikipedia.org/wiki/Charles_Dickens"}},
      {{"atomic_fact": "Charles Dickens wrote 'A Tale of Two Cities' in 1859", "url": "https://en.wikipedia.org/wiki/Charles_Dickens"}},
      {{"atomic_fact": "'A Tale of Two Cities' was written in 1859", "url": "https://en.wikipedia.org/wiki/Charles_Dickens"}},
      {{"atomic_fact": "Charles Dickens was 47 years old when he wrote it", "url": null}}
    ]

    \n"""

ATOMIC_FACTS_EXTRACTION_PROMPT_UPDATED = """
You are an expert NLP data-extraction system. Your task is to decompose an essay into standardized, fully self-contained "atomic facts" for evaluation by an academic expert. The essay is written by an LLM in response to a given question.

### INPUT FORMAT
You will receive:
- <question>: The prompt or question being answered.
- <answer>: The LLM's response.

### EXTRACTION RULES
1. **Decompose to Atomic Units**: Break the text down into its simplest standalone propositions. An atomic fact must contain only a single idea. If one sentence contains two separate ideas, then produce two atomic facts. An idea here might be a fact, a recommendation or an opinion.
2. **Contextualize (Self-Containment)**: Resolve all pronouns (he, she, it, they), relative pronouns (which, that), and implicit references using context from surrounding sentences and from the question. Every fact must be fully understandable on its own without reading the original essay.
3. **Preserve URLs**: If a sentence contains a URL citation, attach that exact URL to every atomic fact derived from that sentence. If no URL is present, assign `null`. Do not modify or sanitize the URL string. If there is more than one URL, attach whichever one is most associated with the atom.
4. **Strict Truthfulness to Source**: Do not add outside knowledge, correct errors, infer unstated points, or alter the meaning of the source text.
5. **Preserve Modality and Qualifiers**: Retain all words that express degree, probability, frequency, or certainty (e.g., always, usually, sometimes, maybe, might, never, strictly, possibly). Dropping these modifiers alters the truth value and scope of the statement. Keep the context to make it clear who is making each claim.
6. **Preserve logic**: If one idea is spread across two sentences in the text, it may be best expressed as a single atomic fact. Atomic facts correspond to ideas, not sentences.
7. **Preserve emphasis**: If the text begins with a direct answer to the question, keep that in the first atomic fact. For example, if the text starts "Yes, it will rain", then the corresponding atom should be "Yes, it will rain" in order to retain the emphasis. In the same way, keep words like "no", "sometimes" and "it depends".

### OUTPUT FORMAT
Return strictly valid JSON matching the schema below. Do not include markdown text, preamble, or conversational notes outside the JSON array.

[
  {
    "atomic_fact": "String",
    "url": "String or null"
  }
]


### EXAMPLE

The question is '''Who wrote 'A Tale of Two Cities' and when?'''

and the provided answer is:
A Tale of Two Cities was written by Charles Dickens in 1859 [https://en.wikipedia.org/wiki/Charles_Dickens]. It is set in London and Paris before and during the French Revolution. The opening line is "It was the best of times, it was the worst of times." Dickens was 47 years old when he wrote it.

Output:
[
  {"atomic_fact": "A Tale of Two Cities was written by Charles Dickens.", "url": "https://en.wikipedia.org/wiki/Charles_Dickens"},
  {"atomic_fact": "'A Tale of Two Cities' was written in 1859.", "url": "https://en.wikipedia.org/wiki/Charles_Dickens"},
  {"atomic_fact": "'A Tale of Two Cities' is set in London and Paris.", "url": null},
  {"atomic_fact": "'A Tale of Two Cities' is set before and during the French Revolution.", "url": null},
  {"atomic_fact": "The opening line of 'A Tale of Two Cities' is \"It was the best of times, it was the worst of times.\"", "url": null},
  {"atomic_fact": "Charles Dickens was 47 years old when he wrote 'A Tale of Two Cities'.", "url": null}
]


### EXAMPLE 2

The question is '''What ID do I need to vote?'''

and the provided answer is:
To vote, you must show your passport or a driving licence. If you've retired, you can show your pension book instead.

Output:
[
  {"atomic_fact": "Voters must show a passport or a driving licence or (if retired) a pension book.", "url": null}
]

### EXAMPLE 3

The question is '''Are vaccines dangerous?'''

and the provided answer is:
No, everyone should be vaccinated.

Output:
[
  {"atomic_fact": "No, vaccines are not dangerous.", "url": null},
  {"atomic_fact": "Everyone should be vaccinated.", "url": null}
]

"""
