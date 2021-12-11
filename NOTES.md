# Helpful Notes on Tutela Documentation

Links that we found helpful in developing and deploying Tutela on EC2. Tutela uses a PostgreSQL database to store clusters.

- See https://torn.community/t/funded-bounty-anonymity-research-tools/1437.
- Built on top of https://github.com/etherclust/etherclust.
- Making service accounts: https://cloud.google.com/docs/authentication/getting-started
- Deployment to EC2: https://www.twilio.com/blog/deploy-flask-python-app-aws
- PostgreSQL on EC2: https://faun.pub/installing-postgresql-in-aws-ubuntu-ec2-instance-b3ecc78caea5
- More PostgreSQL: https://ubiq.co/database-blog/how-to-create-user-with-superuser-privileges-in-postgresql/
- PostgresSQL peer auth: https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge
- Increasing mnt space: https://medium.com/@m.yunan.helmy/increase-the-size-of-ebs-volume-in-your-ec2-instance-3859e4be6cb7
- Ubuntu hacks: https://www.geeksforgeeks.org/setting-python3-as-default-in-linux
- Install gsutil: https://cloud.google.com/storage/docs/gsutil_install
- Install google cloud cmdline: https://cloud.google.com/sdk/docs/install
- Uploading files to S3: https://medium.com/expedia-group-tech/how-to-upload-large-files-to-aws-s3-200549da5ec1
- Flask deployment: https://medium.com/innovation-incubator/deploy-a-flask-app-on-aws-ec2-d1d774c275a2
- Redis config: https://redis.io/topics/config
- Redis quickstart: https://redis.io/topics/quickstart
- Installing SSL: https://linuxbeast.com/tutorials/aws/install-lets-encrypt-with-apache2-on-ec2-ubuntu-18-04/

## Usage 

Add code to your python path:

```
source init_env.sh
```

## Logs

[10.28.21] Added Redis to cache queries, so we don't need to continually bug firebase. However, this is only a bandaid as this does not support fast querying.

[10.29.21] Switching to an SQL model for fast queries. Removes need for firebase and Redis.

[11.9.21] Important to keep uploading to db vs creating clusters files as separate. We also assume that the graph will be small enough to fit into memory. Some doubt here.

## Computing clusters

Here is the procedure of steps to compute "deposit address reuse" clusters. This involves processing CSVs of > 1 Tb, so care must be taken to not exceed constraints on storage and RAM.

0. Download raw block and transaction data using `scripts/dl_bucket.py`.
1. Sort via external merge sort (see `scripts/sort_big_csv.py`).
2. Run `scripts/run_deposit.py` to generate `data.csv` and `metadata.csv`.
3. Prune `data.csv` -> `data-pruned.csv` using `scripts/prune_data.csv`.
4. Prune `metadata.csv` -> `metadata-pruned.csv` using `scripts/metadata-pruned.csv`.
5. Run `scripts/run_nx.py` to generate `user_clusters.json` and `exchange_clusters.json`.
6. Run `combine_metadata.py` to generate `metadata-final.csv`.
