# Order of Operations

1. Run `run_deposit.py` to generate a `metadata.csv` and `data.csv` file. 
2. Run `prune_data.py` to generate a `data-pruned.csv` file.
3. Run `prune_metadata.py` to generate a `metadata-pruned.csv` file.
4. Run `tornadocash/run_exact_match_heuristic.py` to generate `exact_match_clusters.json` and `exact_match_tx2addr.json` files.
5. Run `tornadocash/run_gas_price_heuristic.py` to generate `gas_price_clusters_by_pool.json`, `gas_price_tx2addr_by_pool.json`, `gas_price_address_set_by_pool.json`, and `gas_price_metadata_by_pool.csv` files.
6. Run `tornadocash/run_same_num_txs_heuristic.py` to generate `same_num_txs_clusters.json`, `same_num_txs_tx2addr.json`, `same_num_txs_address_set.json`, and `same_num_txs_metadata.csv` files.
7. Run `heuristic_metadata.py` to generate `metadata-joined.csv`.
8. Run `run_nx.py` to generate `metadata-final.csv`. This is the file that will be used to populate the PostgreSQL database.