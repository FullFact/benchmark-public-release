# Benchmark Analysis Pipeline

This is the code which runs the benchmark analysis, designed to run once a day.
Its primary goal is to extract and group atoms from the LLM responses collected each day,
and then score each model's responses.

## Pipeline

`runner.run_for_day()` runs four stages for one day:

1. **Atomic fact extraction** (`auto_marker.extract_atoms_one_day`): Gemini breaks each
   response into short, self-contained "atomic" facts (prompt in `prompts.py`). Named
   entities are tagged with GLiNER (`api/ner.py`).
2. **Deduplication** (`consistency.find_repeated_atoms`, `deduplication.py`): each new
   atom is compared with earlier atoms for the same question. A fast bi-encoder finds
   candidates. Near-identical text is accepted outright. Other candidates must agree
   on entities and numbers (`parse_numbers.py`) and then pass a cross-encoder check. Each atom
   is linked to its earliest match, preferring matches from the same model.
3. **Source extraction** (`source_analysis.py`): URLs cited in each response.
4. **Scoring** (`auto_marker.py`, `consistency.py`): atoms are matched against
   human-annotated marking schemes. This gives factual, civics, timeliness, transparency
   and consistency scores.

## Data access

In production, the pipeline reads responses from the llm-polygraph API and reads and
writes atoms, marking schemes and scores in a Cloud SQL database. In this release,
`api/service.py` is a stand-in with the same interface. It reads the public SQLite
export, read-only, and keeps anything the pipeline would write in memory.

## Running

From the repository root, with the export unzipped into `data/`:

```sh
uv sync
uv run polygraph-benchmark --start-date 2026-09-22 --output scores.json
```

To use a different export, set `BENCHMARK_DB`. The export already contains atoms for
every response, so extraction is skipped. 
Running it needs GEMINI_API_KEY to extract atoms.
