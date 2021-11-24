import numpy as np
import pandas as pd
from typing import Dict, Any
from sqlalchemy import desc, cast, Float
from app.models import Address

# CONSTS for schema
ADDRESS_COL = 'address'
ENTITY_COL = 'entity'
CONF_COL = 'confidence'
NAME_COL = 'name'
EOA = 'eoa'
DEPOSIT = 'deposit'
EXCHANGE = 'exchange'
DEX = 'dex'
DEFI = 'defi'
ICO_WALLET = 'ico wallet'
MINING = 'mining'


def safe_int(x, default=0):
    try: 
        return int(x)
    except:
        return default


def get_order_command(s, descending):
    s = s.strip()
    column = None
    if s == CONF_COL:
        column = cast(Address.conf, Float)
    elif s == ENTITY_COL:
        column = Address.entity
    elif s == NAME_COL:
        column = Address.name
    elif s == ADDRESS_COL:
        column = Address.address 
    else: 
        column = Address.id # default 

    if descending:
        return desc(column)

    return column # ascending


def get_anonymity_score(
    cluster_confs: np.array,
    cluster_sizes: np.array,
    slope: float = 1,
) -> float:
    """
    Since we are still at one heuristic, let's compute the anonymity
    score as 1 - np.tanh(slope * cluster_conf * cluster_size) where 
    slope is a hyperparameter controlling the slope of the TanH.
    """
    return 1 - np.tanh(slope * np.dot(cluster_confs, cluster_sizes))


def get_known_attrs(known_addresses: pd.DataFrame, address: str) -> Dict[str, Any]:
    """
    Process features from the known_addresses dataframe for a single address.
    """
    result: pd.DataFrame = known_addresses[known_addresses.address == address]
    if len(result) == 0:
        return {}
    result: pd.Series = result.iloc[0]
    result: Dict[str, Any] = result.to_dict()
    del result['address']
    result['legitimacy'] = result['label']
    del result['label']
    return result


def entity_to_str(i):
    if i == 0:
        return EOA
    elif i == 1:
        return DEPOSIT
    elif i == 2:
        return EXCHANGE
    elif i == 3:
        return DEX
    elif i == 4:
        return DEFI
    elif i == 5:
        return ICO_WALLET
    elif i == 6:
        return MINING
    else:
        raise Exception(f'Fatal error: {i}')


def entity_to_int(s):
    if s == EOA:
        return 0
    elif s == DEPOSIT:
        return 1
    elif s == EXCHANGE:
        return 2
    elif s == DEX:
        return 3
    elif s == DEFI:
        return 4
    elif s == ICO_WALLET:
        return 5
    elif s == MINING:
        return 6
    else:
        raise Exception(f'Fatal error: {s}')
