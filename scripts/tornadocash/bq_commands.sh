#!/bin/bash

# load contract table into bigquery (to be run once)
bq load lexical-theory-329617:tornado_transactions.tornadocontracts ./data/static/tcash/tornadocontracts.csv ./data/static/tcash/tornadocontracts_schema.json
# make (empty) traces and transactions table
bq mk --schema ./data/static/tcash/traces_schema.json lexical-theory-329617:tornado_transactions.traces
bq mk --schema ./data/static/tcash/transactions_schema.json lexical-theory-329617:tornado_transactions.transactions

# make tornado traces table
bq query --destination_table lexical-theory-329617:tornado_transactions.traces --use_legacy_sql=false 'select * from bigquery-public-data.crypto_ethereum.traces where (to_address in (select address from lexical-theory-329617.tornado_transactions.tornadocontracts)) and substr(input, 1, 10) in ("0xb214faa5", "0x21a0adb6")'
# make tornado transactions table
bq query --destination_table lexical-theory-329617:tornado_transactions.transactions --use_legacy_sql=false 'select * from bigquery-public-data.crypto_ethereum.transactions as b where b.hash in (select transaction_hash from lexical-theory-329617.tornado_transactions.traces)'
