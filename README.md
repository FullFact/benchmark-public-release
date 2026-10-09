# benchmark-public-release
A public repository for sharing code and data from the Full Fact AI Trust Benchmark

## Report 1 (October 2026)

All the data discussed in Full Fact's first benchmark report can be found in the `report-1/data` folder.

### Installation

`cd report-1/data`

`unzip benchmark-2026-07-13-to-2026-09-22-filtered.sqlite.zip`

to exctact `data/benchmark-2026-07-13-to-2026-09-22-filtered.sqlite`

`cd report-1/`

Add your Gemini API key to the environment:

`export GEMINI_API_KEY=<insert your key here>`

Run the main demo script:

`uv run python scripts/demo.py`

You should then see the list of questions we analysed; an example of a chatbot response; and (if you have access to the Gemini API), a list of extracted atoms. 