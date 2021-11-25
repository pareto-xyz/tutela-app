import os
import redis
import pandas as pd
from typing import Any, Dict
from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from ens import ENS
from web3 import Web3

app: Any = Flask(__name__)
app.config.from_object(Config)
db: SQLAlchemy = SQLAlchemy(app)  # init SQLite DB
migrate: Migrate = Migrate(app, db)
rds = redis.Redis(host='localhost', port=6380, db=0)


class InfuraAuth:

    def __init__(self):
        self._id: str = os.environ['INFURA_ALPHA_ID']
        self._secret: str = os.environ['INFURA_ALPHA_SECRET']

    def auth(self) -> Dict[str, Any]:
        return dict(id=self._id, secret=self._secret)


def get_w3(infura: InfuraAuth) -> Web3:
    auth_dict: Dict[str, Any] = infura.auth()
    uri: str = f'https://:{auth_dict["secret"]}@mainnet.infura.io/v3/{auth_dict["id"]}'
    w3 = Web3(Web3.HTTPProvider(uri))
    return w3


def get_ens(w3: Web3) -> ENS:
    return ENS.fromWeb3(w3)

infura: InfuraAuth = InfuraAuth()
w3: Web3 = get_w3(infura)
ns: ENS = get_ens(w3)


def get_realpath(path: str) -> str:
    base_dir: str = os.path.dirname(__file__)
    path: str = os.path.join(base_dir, path)
    return os.path.realpath(path)


known_addresses: pd.DataFrame = pd.read_csv(
    get_realpath('static/data/known_addresses.csv'))


from app import views, models
