"""
Contains utilities to download recent data needed to run heuristics
on tornado cash data.

Most of the heuristics require access to all of tornado cash at once.
As such, we cannot just download a daily slice (an increment) and 
process that. The best we can do is the following:

1) Keep a table of Tornado Cash trace and transaction data in BigQuery. 
We will update this table daily. 

2) Once updated, the table will be exported a Google cloud bucket. We 
will delete prior contents in bucket before exporting.

3) Download the trace and transactions buckets locally and rename to 
expected tornado_trace.csv and tornado_transaction.csv.

4) Run the preprocessing script to generate complete_withdraw_tx.csv
and complete_deposit_tx.csv. 

5) Save to disk (overwrite old files).
"""
