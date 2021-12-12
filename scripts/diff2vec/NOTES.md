# Order of Operations

1. Run `make_graph.py` to generate `graph-raw.csv` (40 Gb)
2. Run `compress_graph.py` to generate `graph-address.json` (7 Gb) and `graph-compressed.csv` (7 Gb)
3. Run `run_diff2vec.py` with the two files in (2) as input.
