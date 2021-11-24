import math
import json
import numpy as np
from typing import Dict, Optional, List, Any, Set, Tuple

from app import app, w3, ns, known_addresses
from app.models import Address, ExactMatch, GasPrice
from app.utils import \
    safe_int, get_anonymity_score, get_order_command, \
    entity_to_int, entity_to_str, \
    ADDRESS_COL, NAME_COL, ENTITY_COL, CONF_COL, \
    EOA, DEPOSIT, EXCHANGE, DEX, DEFI, ICO_WALLET, MINING
from app.lib.w3 import query_web3

from flask import request, Response
from flask import render_template
from sqlalchemy import or_

from app.utils import get_known_attrs

PAGE_LIMIT = 50
HARD_MAX: int = 100000


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['GET'])
def cluster():
    # return render_template('cluster.html')
    return render_template('maintenance.html')


@app.route('/search', methods=['GET'])
def search():
    address: str = request.args.get('address', '')
    address: str = address.lower()  # important to get lower case

    # --- pagination info ---
    page: int = safe_int(request.args.get('page', 0), 0)
    size: int = safe_int(request.args.get('limit', PAGE_LIMIT), PAGE_LIMIT)
    size: int = min(size, PAGE_LIMIT)

    table_cols: Set[str] = set(Address.__table__.columns.keys())

    # user can sort by column
    sort_by: str = request.args.get('sort', ENTITY_COL)
    if not request.args.get('sort'):
        desc_sort = False  # default is entity asc
    else:
        desc_sort: bool = bool(
            request.args.get(
                'descending',
                False,
                type=lambda v: v.lower() != 'false',
            )
        )

    # --- user can filter by columns (* means everything is supported) ---
    filter_min_conf: float = float(request.args.get('filter_min_' + CONF_COL, 0))
    filter_max_conf: float = float(request.args.get('filter_max_' + CONF_COL, 1))
    filter_entity: str = request.args.get('filter_' + ENTITY_COL, '*')
    filter_name: str = request.args.get('filter_' + NAME_COL, '*')

    filter_by: List[Any] = []
    # A bit of sanity checking for confidences.
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


    def to_dict(
        addr: Address, 
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

        for k in to_remove:
            if k in output:
                del output[k]
        for key, func in to_transform:
            output[key] = func(output[key])
        return output

    def is_valid(obj):
        return (obj is not None)  # and (obj != -1)

    def compute_anonymity_score(
        addr: Address,
        exchange_weight: float = 0.1,
        slope: float = 0.1,
        ) -> float:
        """
        Only EOA addresses have an anonymity score. If we get an exchange,
        we return an anonymity score of 0. If we get a deposit, we return -1,
        which represents N/A.

        For EOA addresses, we grade the anonymity by the confidence and number 
        of other EOA addresses in the same cluster, as well as the confidence 
        and number of other exchanges in the same cluster (which we find through
        the deposits this address interacts with). Exchange interactions are 
        discounted (by `exchange_weight` factor) compared to other EOAs.
        """
        if addr.entity == entity_to_int(DEPOSIT):
            return -1  # represents N/A
        elif addr.entity == entity_to_int(EXCHANGE):
            return 0   # CEX have no anonymity

        assert addr.entity == entity_to_int(EOA), \
            f'Unknown entity: {entity_to_str(addr.entity)}'

        cluster_confs: List[float] = []
        cluster_sizes: List[float] = []

        if addr.user_cluster is not None:
            # find all other EOA addresses with same `dar_user_cluster`.
            num_cluster: int = Address.query.filter(
                Address.user_cluster == addr.user_cluster,
                or_(Address.entity == entity_to_int(EOA)),
            ).limit(HARD_MAX).count()
            cluster_confs.append(addr.conf)
            cluster_sizes.append(num_cluster)

            # find all DEPOSIT address with same `user_cluster`.
            deposits: Optional[List[Address]] = Address.query.filter(
                Address.user_cluster == addr.user_cluster,
                Address.entity == entity_to_int(DEPOSIT),
            ).limit(HARD_MAX).all()

            exchanges: Set[str] = set([
                deposit.exchange_cluster for deposit in deposits])
            cluster_confs.append(addr.conf * exchange_weight)
            cluster_sizes.append(len(exchanges))

        cluster_confs: np.array = np.array(cluster_confs)
        cluster_sizes: np.array = np.array(cluster_sizes)
        score: float = get_anonymity_score(
            cluster_confs, cluster_sizes, slope = slope)

        return score

    def query_exact_match_heuristic(address: str) -> Dict[str, Any]:
        """
        Given an address, find out how many times this address' txs
        appear in a exact match heuristic.
        """
        rows: Optional[List[ExactMatch]] = \
            ExactMatch.query.filter_by(address = address).all()

        history: Dict[str, List[Dict[str, str]]] = []
        if (len(rows) > 0):
            for row in rows:
                # find cluster for this transaction (w/ this address)
                cluster: List[ExactMatch] = ExactMatch.query.filter_by(
                    cluster = row.cluster).all()
                cluster: List[Dict[str, str]] = [
                    dict(address=member.address, transaction=member.transaction)
                    for member in cluster
                ]
                history[row.transaction] = cluster

        return dict(reveal_size=len(history), history=history)

    def query_gas_price_heuristic(address: str) -> Dict[str, Any]:
        """
        Given an address, find out how many times this address' txs 
        appears in a same gas price reveal. We will return the tx info too?
        """
        rows: Optional[List[GasPrice]] = \
            GasPrice.query.filter_by(address = address).all()
        history: Dict[str, List[Dict[str, str]]] = []
        if len(rows) > 0:
            for row in rows:
                # find cluster for this tx
                cluster: List[GasPrice] = GasPrice.query.filter_by(
                    cluster = row.cluster).all()
                cluster: List[Dict[str, str]] = [
                    dict(address=member.address, transaction=member.transaction)
                    for member in cluster
                ]
                history[row.transaction] = cluster

        return dict(reveal_size=len(history), history=history)

    # --- default response object template ---
    output: Dict[str, Any] = {
        'data': {
            'query': {
                'address': address, 
                'metadata': {}, 
                'tornado': {
                    'exact_match': {},
                    'gas_price': {},
                }
            },
            'cluster': [],
            'metadata': {
                'cluster_size': 0,
                'num_pages': 0,
                'page': page,
                'limit': size,
                'filter_by': {
                    'min_conf': filter_min_conf,
                    'max_conf': filter_max_conf,
                    'entity': filter_entity,
                    'name': filter_name,
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
                            EOA, DEPOSIT, EXCHANGE, DEX, DEFI, ICO_WALLET, MINING],
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
    }

    if len(address) > 0:
        offset: int = page * size

        # --- search for address ---
        addr: Optional[Address] = Address.query.filter_by(address = address).first()

        if addr is not None:  # make sure address exists
            entity: str = entity_to_str(addr.entity)
            addr_metadata: Dict[str, Any] = json.loads(addr.meta_data)  # load metadata
            output['data']['query']['metadata'] = addr_metadata

            # store the clusters in here
            cluster: List[Address] = []
            # stores cluster size with filters. This is necessary to reflect changes
            # in # of elements with new filters.
            cluster_size: int = 0

            query_data: Dict[str, Any] = output['data']['query']
            output['data']['query'] = {
                **query_data, 
                **to_dict(addr, to_transform=[('entity', entity_to_str)])
            }

            if entity == EOA:
                # --- compute clusters if you are an EOA ---
                if is_valid(addr.user_cluster):
                    order_command: Any = get_order_command(sort_by, desc_sort)

                    # find all deposit/eoa addresses in the same cluster & filtering attrs
                    query_: Any = Address.query.filter(
                        Address.user_cluster == addr.user_cluster,
                        *filter_by
                    )
                    cluster_: Optional[List[Address]] = query_\
                        .order_by(order_command)\
                        .offset(offset).limit(size).all()

                    if cluster_ is not None:
                        cluster_: List[Dict[str, Any]] = [
                            to_dict(
                                c,
                                to_remove=['id'],
                                to_transform=[('entity', entity_to_str)],
                            ) 
                            for c in cluster_
                        ]
                        cluster += cluster_
                        # get total number of elements in query
                        cluster_size_: int = query_.limit(HARD_MAX).count()
                        cluster_size += cluster_size_

            elif entity == DEPOSIT:
                # --- compute clusters if you are a deposit ---
                # for deposits, we can both look up all relevant eoa's and 
                # all relevant exchanges. These are in two different clusters
                if is_valid(addr.user_cluster):
                    order_command: Any = get_order_command(sort_by, desc_sort)

                    query_: Any = Address.query.filter(
                        Address.user_cluster == addr.user_cluster,
                        *filter_by
                    )
                    cluster_: Optional[List[Address]] = query_\
                        .order_by(order_command)\
                        .offset(offset).limit(size).all()
                   
                    if cluster_ is not None:
                        cluster_: Dict[str, Any] = [
                            to_dict(
                                c,
                                to_remove=['id'],
                                to_transform=[('entity', entity_to_str)],
                            )
                            for c in cluster_
                        ]
                        cluster += cluster_
                        cluster_size_: int = query_.limit(HARD_MAX).count()
                        cluster_size += cluster_size_

            elif entity == EXCHANGE:
                # --- compute clusters if you are an exchange ---
                # find all deposit/exchange addresses in the same cluster
                if is_valid(addr.exchange_cluster):
                    order_command: Any = get_order_command(sort_by, desc_sort)

                    query_: Any = Address.query.filter(
                        Address.exchange_cluster == addr.exchange_cluster,
                        *filter_by
                    )
                    cluster_: Optional[List[Address]] = query_\
                        .order_by(order_command)\
                        .offset(offset).limit(size).all()

                    if cluster_ is not None: 
                        cluster_: Dict[str, Any] = [
                            to_dict(
                                c,
                                to_remove=['id'],
                                to_transform=[('entity', entity_to_str)]
                            ) 
                            for c in cluster_
                        ]
                        cluster += cluster_
                        cluster_size_: int = query_.limit(HARD_MAX).count()
                        cluster_size += cluster_size_
            else:
                raise Exception(f'Entity {entity} not supported.')

            output['data']['cluster'] = cluster
            output['data']['metadata']['cluster_size'] = cluster_size
            output['data']['metadata']['num_pages'] = int(math.ceil(cluster_size / size))

            # --- compute anonymity score using hyperbolic fn ---
            anon_score = compute_anonymity_score(addr)
            anon_score: float = round(anon_score, 3)  # brevity is a virtue
            output['data']['query']['anonymity_score'] = anon_score

            # --- query web3 to get current information about this address ---
            web3_resp: Dict[str, Any] = query_web3(address, w3, ns)
            query_metadata: Dict[str, Any] = output['data']['query']['metadata']
            output['data']['query']['metadata'] = {**query_metadata, **web3_resp}

            # --- check if we know existing information about this address ---
            known_lookup: Dict[str, Any] = get_known_attrs(known_addresses, address)
            if len(known_lookup) > 0:
                query_metadata: Dict[str, Any] = output['data']['query']['metadata']
                output['data']['query']['metadata'] = {**query_metadata, **known_lookup}

        # --- check tornado queries ---
        # Note that this is out of the `Address` existence check
        exact_match_dict: Dict[str, Any] = query_exact_match_heuristic(address)
        gas_price_dict: Dict[str, Any] = query_gas_price_heuristic(address)
        output['data']['query']['tornado']['exact_match'] = exact_match_dict
        output['data']['query']['tornado']['gas_price'] = gas_price_dict

        # if `addr` doesnt exist, then we assume no clustering
        output['success'] = 1

    response: str = json.dumps(output)
    return Response(response=response)
