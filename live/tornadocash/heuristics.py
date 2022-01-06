"""
Contains utilities to compute heuristics on all tornado cash data.

1) This assumes get_data.py has been run. It will need access to 
updated files complete_withdraw_tx.csv and complete_deposit_tx.csv. 

2) Run the following heuristics in order:
    - ExactMatch
    - UniqueGasPrice
    - SameNumTx
    - LinkedTx
    - TornMine

3) Run json_to_sql_format functions to generate a processed.csv for 
each heuristic. We will not write any other files to disk.

4) Delete existing content in db. Insert new CSV files into db.
If possible look into ovewriting here rather than deleting rows.
"""