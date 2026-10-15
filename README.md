# benchmark-public-release
A public repository for sharing code and data from the Full Fact AI Trust Benchmark. The report itself can be found at [fullfact.org/ai-benchmark](https://fullfact.org/ai-benchmark/).


## Report 1 (October 2026)

All the data discussed in Full Fact's [first Public Interest AI Trust Benchmark report](https://fullfact.org/ai-benchmark/report-one/) can be found in the SQLite database included in this repository. The code here is a subset of the code we used internally to process the data, including extracting atomic facts, deduplicating them and scoring chatbot responses. To keep things simple, we haven't included the browser-based UI and we've moved the database to local SQLite storage instead of a shared cloud SQL database. 

For subsequent versions of the benchmark, we have already developed a more robust version of the tool with better handling of atomic facts and improved scoring methods. This version reflects how the data and analysis of the first report was produced. It is not intended as a fully working system nor is the code being maintained.

For each of the 22 questions we studied, members of Full Fact's Editorial team wrote reference answers, describing what information should be contained in each response. These can be found in the [reference_answers](report-1/reference_answers/) folder.

### Installation

After cloning the repo,  you need to decompress the archived data set:

`cd report-1/data`

`unzip benchmark-2026-07-13-to-2026-09-22-filtered.sqlite.zip`

to extract `data/benchmark-2026-07-13-to-2026-09-22-filtered.sqlite`


Now you can run the first demo script:

`cd report-1/`

`uv run python scripts/demo_data_reports.py`

You will then see the list of questions we analysed and an example of a chatbot response. The data is read from the local archived sqlite database.

A second script extracts atoms from a sample response by calling Gemini. To run this, first add your Gemini API key to the environment:

`export GEMINI_API_KEY=<insert your key here>`

Then run the extraction demo script:

`uv run python scripts/demo_extract.py`

If you have access to the Gemini API, you should see a list of extracted atoms. Due to the non-deterministic nature of LLMs, you can expect a slightly different list each time.

During the annotation period, the `run_for_day()` in `runner.py` was executed every day after the chatbot responses had been gathered. The resulting atomic facts, groups and source URLs were shown to the annotators in a simple UI.
