# Web API through Flask

Before you run the project locally, follow the installation steps below. 

After that, do the following:

### from the overall project directory: 
```
sh start_redis.sh       (start the redis server)
```

### in a new terminal tab:
```
cd webapp
python run.py           (run the actual app) 
```

Note: if it throws an error about `ens` not being found, do NOT install `ens`. install/reinstall `web3` (`pip install web3`).
If it throws other import errors, then just `pip install {package-name}`

## Usage + setup 

You will need to run for every fresh terminal instance

```
export FLASK_APP=run.py
export INFURA_ALPHA_ID='your-infura-alpha-id'
export INFURA_ALPHA_SECRET='your-infura-alpha-secret'
source init_env.sh
```
If you get issues later about webapp.run not being found, then replace `run.py` with the absolute path of webapp/run.py

Note that you will need to do this even if you did this in the parent directory.

###  Dependencies

create a venv using `python -m venv ./path/to/where/you/want/your/venv`
then: `source ./{path-to-venv}/bin/activate`

then 
```
cd webapp 
pip -r install requirements.txt
```


Redis installations

```
sudo apt install redis-server
sudo apt install redis-cli
sudo apt install redis-tools
```

Flask db setup:

```
flask db init
flask db migrate
flask db upgrade
```

## other useful commands during development 

### Database 
to update the database (if more / fewer tables exist)

```
cd webapp/app
rm -rf migrations/  

psql -U tornado                         (open psql cli)
\c tornado;                             (connect to the tornado database)
\dt;                                    (to see current tables) 
drop table {name of table}              (to delete table so that you can repopulate later)
\q                                      (quit)

flask db init
flask db migrate
flask db upgrade
```

### redis 
If you made backend changes that change the api response, you'll need to flush redis. 

`redis-cli -h localhost -p 6380`
^ make sure redis-server is running in another terminal tab simultaneously. 

### if changing front-end code (involving react):

`cd webapp/app/static`
`npm run watch`

now, whenever you change and save your code, it will automatically update the running front-end code. 

