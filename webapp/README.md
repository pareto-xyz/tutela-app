# Web API through Flask

To run, use `python run.py`. 

## Usage

You will need to run for every fresh terminal instance

```
export FLASK_APP=run.py
source init_env.sh
```

Note that you will need to do this even if you did this in the parent directory.

## Dependencies

```
psycopg2-binary
flask-sqlalchemy
flask-migrate
flask
```

Redis installations

```
sudo apt install redis-server
sudo apt install redis-cli
sudo apt install redis-tools
```

## Helpful Commands

```
Dump database

pg_dump the_db_name > the_backup.sql

Restore database

psql the_new_dev_db < the_backup.sql
```

## Flask commands

```
flask db init
flask db migrate 
flask db upgrade
```
