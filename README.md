# benchmark-public-release
A public repository for sharing code and data from the Full Fact AI Trust Benchmark

## Report 1 (October 2026)

All the data discussed in Full Fact's first benchmark report can be found in the SQLite database included in this repository. The code here is a subset of the code we used internally to process the data, including extracting atomic facts, deduplicating them and scoring chatbot responses. To keep things simple, we haven't included the browser-based UI and we've moved the database to local SQLite storage instead of a shared cloud SQL database. 

For subsequent versions of the benchmark, we have already developed a more robust version of the tool with better handling of atomic facts and improved scoring methods. This version reflects how the data and analysis of the first report was produced. It is not intended as a fully working system nor is the code being maintained.

### Installation

After cloning the repo,  you need decompress the archived data set.

`cd report-1/data`

`unzip benchmark-2026-07-13-to-2026-09-22-filtered.sqlite.zip`

to extract `data/benchmark-2026-07-13-to-2026-09-22-filtered.sqlite`

`cd report-1/`

Now you can run the first demo script:

`uv run python scripts/demo_data_reports.py`

You should then see the list of questions we analysed and an example of a chatbot response. The data is read from the local archived sqlite database.

The second script extracts atoms from a sample response by calling Gemini.

Add your Gemini API key to the environment:

`export GEMINI_API_KEY=<insert your key here>`

Run the extraction script:

`uv run python scripts/demo_extract.py`

If you have access to the Gemini API, you should see a list of extracted atoms. 