# get txs from 1 week
bq --location=US query --destination_table lexical-theory-329617:crypto_ethereum.transactions_1week --use_legacy_sql=false "select from_address, to_address, value, block_timestamp, block_number from bigquery-public-data.crypto_ethereum.transactions where (block_number >= 13285098) and (block_number <= 13330090)"

# get txs from 1 month
bq --location=US query --destination_table lexical-theory-329617:crypto_ethereum.transactions_1month --use_legacy_sql=false "select from_address, to_address, value, block_timestamp, block_number from bigquery-public-data.crypto_ethereum.transactions where (block_number >= 13136427) and (block_number <= 13330090)"

# get txs from 1 year
bq --location=US query --destination_table lexical-theory-329617:crypto_ethereum.transactions_1year --use_legacy_sql=false "select from_address, to_address, value, block_timestamp, block_number from bigquery-public-data.crypto_ethereum.transactions where (block_number >= 10966874) and (block_number <= 13330090)"

# get all txs before adte
bq --location=US query --destination_table lexical-theory-329617:crypto_ethereum.transactions --use_legacy_sql=false "select from_address, to_address, value, block_timestamp, block_number from bigquery-public-data.crypto_ethereum.transactions where block_number <= 13330090"

