import json
import numpy as np
import pandas as pd
from copy import copy
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from sqlalchemy import desc, cast, Float
from app.models import Address, TornadoPool
from sqlalchemy import or_, and_

# CONSTS for schema
ADDRESS_COL: str = 'address'
ENTITY_COL: str = 'entity'
CONF_COL: str = 'confidence'
NAME_COL: str = 'name'
HEURISTIC_COL: str = 'heuristic'
# --
EOA: str = 'eoa'
DEPOSIT: str = 'deposit'
EXCHANGE: str = 'exchange'
DEX: str = 'dex'
DEFI: str = 'defi'
ICO_WALLET: str = 'ico wallet'
MINING: str = 'mining'
TORNADO: str = 'tornado'
# --
GAS_PRICE_HEUR: str = 'unique_gas_price'
DEPO_REUSE_HEUR: str = 'deposit_address_reuse'
SAME_NUM_TX_HEUR: str = 'multi_denomination'
SAME_ADDR_HEUR: str = 'address_match'


def safe_int(x, default=0):
    try: 
        return int(x)
    except:
        return default


def safe_float(x, default=0):
    try: 
        return float(x)
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

def get_display_aliases() -> Dict[str, str]:
    return {
        'num_deposit': 'deposits',
        'num_withdraw': 'withdraws',
        'num_compromised': 'compromised',
        'num_compromised_exact_match': 'address match',
        'num_compromised_gas_price': 'unique gas price',
        'num_compromised_multi_denom': 'multi-denom',
        'conf': 'confidence score',
        'entity': 'address type',
        'balance': 'ETH balance',
        'ens_name': 'ENS',
        'exchange_address': 'exchange address',
        'exchange_name': 'associated exchange',
        'account_type': 'category',
        'label': 'legitimacy',
        'tags': 'other',
        'num_deposits': 'total equal user deposits',
        'compromised': 'compromised deposits',
        'exact_match': 'address match reveals',
        'multi_denom': 'multi-denom reveals',
        'gas_price': 'unique gas price reveals'
    }


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
    if 'label' in result and 'legitimacy' not in result:
        result['legitimacy'] = result['label']
        del result['label']
    return result


def entity_to_str(i: int) -> str:
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
    elif i == 7:
        return TORNADO
    else:
        raise Exception(f'Fatal error: {i}')


def entity_to_int(s: str) -> int:
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
    elif s == TORNADO:
        return 7
    else:
        raise Exception(f'Fatal error: {s}')


def heuristic_to_str(s: int) -> str:
    if s == 0:
        return DEPO_REUSE_HEUR
    elif s == 1:
        return SAME_ADDR_HEUR
    elif s == 2:
        return GAS_PRICE_HEUR
    elif s == 3:
        return SAME_NUM_TX_HEUR
    else:
        raise Exception(f'Fatal error: {s}')


def heuristic_to_int(s: str) -> int:
    if s == DEPO_REUSE_HEUR:
        return 0
    elif s == SAME_ADDR_HEUR: 
        return 1
    elif s == GAS_PRICE_HEUR:
        return 2
    elif s == SAME_NUM_TX_HEUR:
        return 3
    else:
        raise Exception(f'Fatal error: {s}')


def to_dict(
    addr: Address, 
    table_cols: List[str], 
    to_add: Dict = {},
    to_remove: List[str] = [], 
    to_transform: List[Tuple[str, Any]] = [],
) -> Dict[str, Any]:
    """
    Convert a raw row into the form we want to send to the frontend.
    for `to_transform`, each element is in the tuple form: 
        (key_to_override_value_of, function to take of the old value)
    """
    output: Dict[str, Any] = {  # ignore cluster columns
        k: getattr(addr, k) for k in table_cols if 'cluster' not in k
    }
    output['conf'] = round(output['conf'], 3)
    del output['meta_data']  # Q: should we keep this?

    for k, v in to_add.items():
        if k not in output:
            output[k] = v

    for k in to_remove:
        if k in output:
            del output[k]
    for key, func in to_transform:
        output[key] = func(output[key])
    return output


def default_address_response() -> Dict[str, Any]:
    output: Dict[str, Any] = {
        'data': {
            'query': {
                'address': '', 
                'metadata': {}, 
            },
            'tornado': {
                'summary': {
                    'address': {
                        'num_deposit': 0,
                        'num_withdraw': 0,
                        'num_compromised': 0
                    }, 
                },
            },
            'cluster': [],
            'metadata': {
                'cluster_size': 0,
                'num_pages': 0,
                'page': 0,
                'limit': 50,
                'filter_by': {
                    'min_conf': 0,
                    'max_conf': 1,
                    'entity': '*',
                    'name': '*',
                },
                'schema': {
                    ADDRESS_COL: {
                        'type': 'string',
                    },
                    CONF_COL: {
                        'type': 'float',
                        'values': [0, 1],
                    },
                    ENTITY_COL: {
                        'type': 'category',
                        'values': [
                            EOA, 
                            DEPOSIT, 
                            EXCHANGE, 
                            DEX, 
                            DEFI, 
                            ICO_WALLET, 
                            MINING,
                            TORNADO,
                        ],
                    },
                    NAME_COL: {
                        'type': 'string',
                    },
                },
                'sort_default': {
                    'attribute': ENTITY_COL,
                    'descending': False
                }
            }
        },
        'success': 0,
        'is_tornado': 0,
    }
    return output


def default_tornado_response() -> Dict[str, Any]:
    output: Dict[str, Any] = {
        'data': {
            'query': {
                'address': '', 
                'metadata': {
                    'amount': 0,
                    'currency': '',
                    'stats': {
                        'num_deposits': 0,
                        'num_compromised': 0,
                        'exact_match': 0,
                        'gas_price': 0,
                        'multi_denom': 0,
                    }
                },
            },
            'compromised': [],
            'metadata': {
                'compromised_size': 0,
                'num_pages': 0,
                'page': 0,
                'limit': 50,
                'schema': {
                    CONF_COL: {
                        'type': 'float',
                        'values': [0, 1]
                    }, 
                    HEURISTIC_COL: {
                        'type': 'category',
                        'values': [
                            DEPO_REUSE_HEUR, 
                            SAME_ADDR_HEUR, 
                            GAS_PRICE_HEUR, 
                            SAME_NUM_TX_HEUR,
                        ],
                    },
                },
                'sort_default': {
                    'attribute': CONF_COL,
                    'descending': True
                }
            },
        },
        'success': 1,
        'is_tornado': 1,
    }
    return output


def is_valid_address(address: str) -> bool:
    address: str = address.lower().strip()

    if len(address) != 42:
        return False

    if len(address) > 2:
        if address[:2] != '0x':
            return False

    if len(address.split()) != 1:
        return False

    return True


class AddressRequestChecker:
    """
    Given a request object with its args, make sure that it is 
    providing valid arguments.
    """
    
    def __init__(
        self,
        request: Any,
        table_cols: List[str],
        entity_key: str = 'entity',
        conf_key: str = 'confidence',
        name_key: str = 'name',
        default_page: int = 0,
        default_limit: int = 50,
    ):
        self._request: Any = request
        self._table_cols: List[str] = table_cols
        self._entity_key: str = entity_key
        self._conf_key: str = conf_key
        self._name_key: str = name_key
        self._default_page: int = default_page
        self._default_limit: int = default_limit

        self._params: Dict[str, Any] = {}

    def check(self):
        return (self._check_address() and
                self._check_page() and
                self._check_limit() and
                self._check_sort_by() and 
                self._check_filter_by())

    def _check_address(self) -> bool:
        """
        Check that the first two chars are 0x and that the string
        is 42 chars long. Check that there are no spaces.
        """
        address: str = self._request.args.get('address', '')
        is_valid: bool = is_valid_address(address)
        if is_valid:  # if not valid, don't save this
            self._params['address'] = address
        return is_valid

    def _check_page(self) -> bool:
        # intentionally only returns True as we don't want to block a user
        # bc of typo on page
        default: int = self._default_page
        page: Union[str, int] = self._request.args.get('page', default)
        page: int = safe_int(page, default)
        page: int = max(page, 0)  # at least 0
        self._params['page'] = page
        return True

    def _check_limit(self) -> bool:
        # intentionally only returns True as we don't want to block a user
        # bc of typo on limit
        default: int = self._default_limit
        limit: Union[str, int] = self._request.args.get('limit', default)
        limit: int = safe_int(limit, default)
        limit: int = min(max(limit, 1), default)  # at least 1
        self._params['limit'] = limit
        return True

    def _check_sort_by(self) -> bool:
        default: str = self._entity_key
        sort_by: str = self._request.args.get('sort', default)
        # make sure column is a support column
        if sort_by not in self._table_cols:
            sort_by: str = default

        if not self._request.args.get('sort'):
            desc_sort = False  # default is entity asc
        else:
            desc_sort: bool = bool(
                self._request.args.get(
                    'descending', False, type=lambda v: v.lower() != 'false'))

        self._params['sort_by'] = sort_by
        self._params['desc_sort'] = desc_sort
        return True

    def _check_filter_by(self) -> bool:
        filter_min_conf: float = \
            self._request.args.get(f'filter_min_{self._conf_key}', 0)
        filter_max_conf: float = \
            self._request.args.get(f'filter_max_{self._conf_key}', 1)
        filter_min_conf: float = safe_float(filter_min_conf, 0)
        filter_max_conf: float = safe_float(filter_max_conf, 1)

        filter_entity: str = \
            self._request.args.get(f'filter_{self._entity_key}', '*')

        if filter_entity not in [
            EOA, DEPOSIT, EXCHANGE, DEX, DEFI, ICO_WALLET, MINING]:
            filter_entity: str = '*'

        filter_name: str = \
            self._request.args.get(f'filter_{self._name_key}', '*')

        filter_by: List[Any] = []

        # the below will fail if address doesn't exist in table
        if Address.query.filter_by(address = self._params['address']).first():
            if ((filter_min_conf >= 0 and filter_min_conf <= 1) and
                (filter_max_conf >= 0 and filter_max_conf <= 1) and
                (filter_min_conf <= filter_max_conf)):
                filter_by.append(Address.conf >= filter_min_conf)
                filter_by.append(Address.conf <= filter_max_conf)
            if filter_entity != '*':
                filter_by.append(Address.entity == entity_to_int(filter_entity))
            if filter_name != '*': # search either
                filter_by.append(
                    or_(
                        Address.name.ilike('%'+filter_name.lower()+'%'), 
                        Address.address.ilike(filter_name.lower()),
                    ),
                )

        self._params['filter_by'] = filter_by
        self._params['filter_min_conf'] = filter_min_conf
        self._params['filter_max_conf'] = filter_max_conf
        self._params['filter_entity'] = filter_entity
        self._params['filter_name'] = filter_name
        return True

    def get(self, k: str) -> Optional[Any]:
        return self._params.get(k, None)

    def to_str(self):
        _repr: Dict[str, Any] = copy(self._params)
        del _repr['filter_by']
        return json.dumps(_repr, sort_keys=True)


class TornadoPoolRequestChecker:

    def __init__(self, request: Any, default_page: int = 0, default_limit: int = 50):
        self._request: Any = request
        self._default_page: int = default_page
        self._default_limit: int = default_limit
        self._params: Dict[str, Any] = {}

    def check(self):
        return self._check_address() and self._check_page() and self._check_limit()

    def _check_address(self) -> bool:
        """
        Check that the first two chars are 0x and that the string
        is 42 chars long. Check that there are no spaces.
        """
        address: str = self._request.args.get('address', '')
        is_valid: bool = is_valid_address(address)
        if is_valid:  # if not valid, don't save this
            self._params['address'] = address
        return is_valid

    def _check_page(self) -> bool:
        # intentionally only returns True as we don't want to block a user
        # bc of typo on page
        default: int = self._default_page
        page: Union[str, int] = self._request.args.get('page', default)
        page: int = safe_int(page, default)
        page: int = max(page, 0)  # at least 0
        self._params['page'] = page
        return True

    def _check_limit(self) -> bool:
        # intentionally only returns True as we don't want to block a user
        # bc of typo on limit
        default: int = self._default_limit
        limit: Union[str, int] = self._request.args.get('limit', default)
        limit: int = safe_int(limit, default)
        limit: int = min(max(limit, 1), default)  # at least 1
        self._params['limit'] = limit
        return True

    def get(self, k: str) -> Optional[Any]:
        return self._params.get(k, None)

    def to_str(self):
        _repr: Dict[str, Any] = copy(self._params)
        return json.dumps(_repr, sort_keys=True)

# -- Tornado pool utilities --

def is_tornado_address(address: str) -> bool:
    return TornadoPool.query.filter_by(pool = address).count() > 0


def get_equal_user_deposit_txs(address: str) -> Set[str]:
    rows: List[TornadoPool] = \
        TornadoPool.query.filter_by(pool = address).all()
    txs: List[str] = [row.transaction for row in rows]
    return set(txs)


def find_reveals(transactions: List[str], class_: Any) -> Set[str]:
    rows: List[class_] = \
        class_.query.filter(class_.transaction.in_(transactions)).all()
    clusters: List[int] = list(set([row.cluster for row in rows]))
    rows: List[class_] = \
        class_.query.filter(class_.cluster.in_(clusters)).all()

    reveals: List[str] = list(set([row.transaction for row in rows]))
    return set(reveals)
