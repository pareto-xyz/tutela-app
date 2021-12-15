import bz2
import math
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Set

from app import app, w3, ns, rds, known_addresses, tornado_pools
from app.models import \
    Address, ExactMatch, GasPrice, MultiDenom, \
    TornadoDeposit, TornadoWithdraw
from app.utils import \
    get_anonymity_score, get_order_command, \
    entity_to_int, entity_to_str, to_dict, \
    heuristic_to_str, is_valid_address, \
    is_tornado_address, get_equal_user_deposit_txs, find_reveals, \
    AddressRequestChecker, TornadoPoolRequestChecker, \
    default_address_response, default_tornado_response, \
    NAME_COL, ENTITY_COL, CONF_COL, EOA, DEPOSIT, EXCHANGE
from app.lib.w3 import query_web3, get_ens_name, resolve_address

from flask import request, Request, Response
from flask import render_template
from sqlalchemy import or_

from app.utils import get_known_attrs, get_display_aliases

PAGE_LIMIT = 50
HARD_MAX: int = 1000


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['GET'])
def cluster():
    return render_template('cluster.html')


@app.route('/utils/aliases', methods=['GET'])
def alias():
    response: str = json.dumps(get_display_aliases())
    return Response(response=response)


@app.route('/utils/istornado', methods=['GET'])
def istornado():
    address: str = request.args.get('address', '')
    address: str = resolve_address(address, ns)

    output: Dict[str, Any] = {
        'data': {
            'address': address,
            'is_tornado': 1,
            'amount': 0,
            'currency': '',
        },
        'success': 0,
    }

    if not is_valid_address(address):
        return Response(output)

    is_tornado: bool = int(is_tornado_address(address))
    pool: pd.DataFrame = \
        tornado_pools[tornado_pools.address == address].iloc[0]
    amount, currency = pool.tags.strip().split()

    output['data']['is_tornado'] = is_tornado
    output['data']['amount'] = int(amount)
    output['data']['currency'] = currency
    output['success'] = 1

    response: str = json.dumps(output)
    return Response(response)


@app.route('/transaction', methods=['GET'])
def transaction():
    return render_template('transaction.html')


@app.route('/search', methods=['GET'])
def search():
    address: str = request.args.get('address', '')
    # after this call, we should expect address to be an address
    address: str = resolve_address(address, ns)

    # do a simple check that the address is valid
    if not is_valid_address(address):
        return default_address_response()

    # check if address is a tornado pool or not
    is_tornado: bool = is_tornado_address(address)

    if is_tornado:
        # ---------------------------------------------------------
        # MODE #1
        # This is a TCash pool, so we can show specific information
        # about compromised addresses via our heuristics.
        # ---------------------------------------------------------
        response: Response = search_tornado(request)
    else:
        # ---------------------------------------------------------
        # MODE #2
        # This is a regular address, so we can search our dataset
        # for its cluster and complimentary information.
        # ---------------------------------------------------------
        response: Response = search_address(request)

    return response


@app.route('/search/compromised', methods=['GET'])
def haveibeencompromised():
    address: str = request.args.get('address', '')
    pool: str = request.args.get('pool', '')  # tornado pool address
    address: str = resolve_address(address, ns)

    output: Dict[str, Any] = {
        'data': {
            'address': address,
            'pool': pool,
            'compromised_size': 0,
            'compromised': [],
        },
        'success': 0,
    }

    if not is_valid_address(address) or not is_valid_address(pool):
        return Response(json.dumps(output))

    # find all the deposit transactions made by user for this pool
    deposits: Optional[List[TornadoDeposit]] = \
        TornadoDeposit.query.filter_by(
            from_address = address,
            tornado_cash_address = pool,
        ).all()
    deposit_txs: Set[str] = set([d.hash for d in deposits])
    deposit_txs_list: List[str] = list(deposit_txs)

    # search for these txs in the reveal tables
    exact_match_reveals: Set[str] = find_reveals(deposit_txs_list, ExactMatch)
    gas_price_reveals: Set[str] = find_reveals(deposit_txs_list, GasPrice)
    multi_denom_reveals: Set[str] = find_reveals(deposit_txs_list, MultiDenom)

    def format_compromised(
        exact_match_reveals: Set[str],
        gas_price_reveals: Set[str],
        multi_denom_reveals: Set[str],
    ) -> List[Dict[str, Any]]:
        compromised: List[Dict[str, Any]] = []
        for reveal in exact_match_reveals:
            compromised.append({'heuristic': heuristic_to_str(1), 'transaction': reveal})
        for reveal in gas_price_reveals:
            compromised.append({'heuristic': heuristic_to_str(2), 'transaction': reveal})
        for reveal in multi_denom_reveals:
            compromised.append({'heuristic': heuristic_to_str(3), 'transaction': reveal})
        return compromised

    # add compromised sets to response
    compromised: List[Dict[str, Any]] = format_compromised(
        exact_match_reveals, gas_price_reveals, multi_denom_reveals)
    output['data']['compromised'] = compromised
    output['data']['compromised_size'] = len(compromised)
    output['success'] = 1

    response: str = json.dumps(output)
    return Response(response)


def search_address(request: Request) -> Response:
    """
    Master function for serving address requests. This function 
    will first check if the request is valid, then find clusters
    corresponding to this address, as well as return auxilary 
    information, such as web3 info and Tornado specific info.

    Has support for Redis for fast querying. Even if no clusters
    are found, Tornado and basic info is still returned.
    """
    table_cols: Set[str] = set(Address.__table__.columns.keys())

    # Check if this is a valid request searching for an address
    checker: AddressRequestChecker = AddressRequestChecker(
        request,
        table_cols,
        entity_key = ENTITY_COL,
        conf_key = CONF_COL,
        name_key = NAME_COL,
        default_page = 0,
        default_limit = PAGE_LIMIT,
    )
    is_valid_request: bool = checker.check()
    output: Dict[str, Any] = default_address_response()

    if not is_valid_request:   # if not, bunt
        return Response(output)

    address: str = checker.get('address')
    page: int = checker.get('page')
    size: int = checker.get('limit')
    sort_by: str = checker.get('sort_by')
    desc_sort: str = checker.get('desc_sort')
    filter_by: List[Any] = checker.get('filter_by')

    request_repr: str = checker.to_str()

    if rds.exists(request_repr):  # check if this exists in our cache
        response: str = bz2.decompress(rds.get(request_repr)).decode('utf-8')
        return Response(response=response)

    # --- fill out some of the known response fields ---
    output['data']['query']['address'] = address
    output['data']['metadata']['page'] = page
    output['data']['metadata']['limit'] = size
    for k in output['data']['metadata']['filter_by'].keys():
        output['data']['metadata']['filter_by'][k] = checker.get(f'filter_{k}')


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

    def query_heuristic(address: str, class_: Any) -> Set[str]:
        """
        Given an address, find out how many times this address' txs
        appear in a heuristic. Pass the table class for heuristic.
        """
        rows: Optional[List[class_]] = \
            class_.query.filter_by(address = address).all()

        cluster_txs: List[str] = []

        if (len(rows) > 0):
            clusters: List[int] = list(set([row.cluster for row in rows]))
            cluster: List[class_] = \
                class_.query.filter(class_.cluster.in_(clusters)).all()
            cluster_txs: List[str] = [row.transaction for row in cluster]

        return set(cluster_txs)  # no duplicates

    def query_tornado_stats(address: str) -> Dict[str, int]:
        """
        Given a user address, we want to supply a few statistics:

        1) Number of deposits made to Tornado pools.
        2) Number of withdraws made to Tornado pools.
        3) Number of deposits made that are part of a cluster or of a TCash reveal.
        """
        exact_match_txs: Set[str] = query_heuristic(address, ExactMatch)
        gas_price_txs: Set[str] = query_heuristic(address, GasPrice)
        multi_denom_txs: Set[str] = query_heuristic(address, MultiDenom)

        reveal_txs: Set[str] = exact_match_txs.union(gas_price_txs)
        reveal_txs: Set[str] = reveal_txs.union(multi_denom_txs)

        # find all txs where the from_address is the current user.
        deposits: Optional[List[TornadoDeposit]] = \
            TornadoDeposit.query.filter_by(from_address = address).all()
        deposit_txs: Set[str] = set([d.hash for d in deposits])
        num_deposit: int = len(deposit_txs)

        # find all txs where the recipient_address is the current user
        withdraws: Optional[List[TornadoWithdraw]] = \
            TornadoWithdraw.query.filter_by(recipient_address = address).all()
        withdraw_txs: Set[str] = set([w.hash for w in withdraws])
        num_withdraw: int = len(withdraw_txs)

        all_txs: Set[str] = deposit_txs.union(withdraw_txs)
        num_all: int = num_deposit + num_withdraw

        num_remain: int = len(all_txs - reveal_txs)
        num_remain_exact_match: int = len(all_txs - exact_match_txs)
        num_remain_gas_price: int = len(all_txs - gas_price_txs)
        num_remain_multi_denom: int = len(all_txs - multi_denom_txs)

        num_compromised: int = num_all - num_remain
        num_compromised_exact_match = num_all - num_remain_exact_match
        num_compromised_gas_price = num_all - num_remain_gas_price
        num_compromised_multi_denom = num_all - num_remain_multi_denom

        # compute number of txs compromised by TCash heuristics
        stats: Dict[str, int] = dict(
            num_deposit = num_deposit,
            num_withdraw = num_withdraw,
            num_compromised = dict(
                all_reveals = num_compromised,
                num_compromised_exact_match = num_compromised_exact_match,
                num_compromised_gas_price = num_compromised_gas_price,
                num_compromised_multi_denom = num_compromised_multi_denom,
            ),
            num_uncompromised = num_all - num_compromised
        )
        return stats


    if len(address) > 0:
        offset: int = page * size

        # --- search for address ---
        addr: Optional[Address] = Address.query.filter_by(address = address).first()

        if addr is not None:  # make sure address exists
            entity: str = entity_to_str(addr.entity)
            if addr.meta_data is None:
                addr.meta_data = '{}'
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
                **to_dict(addr, table_cols, to_transform=[
                    ('entity', entity_to_str),
                    ('heuristic', heuristic_to_str),
                ])
            }

            if entity == EOA:
                # --- compute clusters if you are an EOA ---
                if addr.user_cluster is not None:
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
                                table_cols,
                                to_add={'ens_name': get_ens_name(c.address, ns)},
                                to_remove=['id'],
                                to_transform=[
                                    ('entity', entity_to_str),
                                    ('heuristic', heuristic_to_str),
                                ],
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
                if addr.user_cluster is not None:
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
                                table_cols,
                                to_add={'ens_name': get_ens_name(c.address, ns)},
                                to_remove=['id'],
                                to_transform=[
                                    ('entity', entity_to_str),
                                    ('heuristic', heuristic_to_str),
                                ],
                            )
                            for c in cluster_
                        ]
                        cluster += cluster_
                        cluster_size_: int = query_.limit(HARD_MAX).count()
                        cluster_size += cluster_size_

            elif entity == EXCHANGE:
                # --- compute clusters if you are an exchange ---
                # find all deposit/exchange addresses in the same cluster
                if addr.exchange_cluster is not None:
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
                                table_cols,
                                to_add={'ens_name': get_ens_name(c.address, ns)},
                                to_remove=['id'],
                                to_transform=[
                                    ('entity', entity_to_str),
                                    ('heuristic', heuristic_to_str),
                                ]
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
        tornado_dict: Dict[str, Any] = query_tornado_stats(address)
        output['data']['tornado']['summary']['address'].update(tornado_dict)

        # if `addr` doesnt exist, then we assume no clustering
        output['success'] = 1

    response: str = json.dumps(output)
    rds.set(request_repr, bz2.compress(response.encode('utf-8')))  # add to cache

    return Response(response=response)


def search_tornado(request: Request) -> Response:
    """
    We know the address we are searching for is a Tornado pool, which
    means we can provide special information about compromises.
    """
    checker: TornadoPoolRequestChecker = TornadoPoolRequestChecker(
        request,
        default_page = 0,
        default_limit = PAGE_LIMIT,
    )
    is_valid_request: bool = checker.check()
    output: Dict[str, Any] = default_tornado_response()

    if not is_valid_request:
        return Response(output)

    # check if we can find in cache
    request_repr: str = checker.to_str()

    if rds.exists(request_repr):  # check if this exists in our cache
        response: str = bz2.decompress(rds.get(request_repr)).decode('utf-8')
        return Response(response=response)

    address: str = checker.get('address')
    page: int = checker.get('page')
    size: int = checker.get('limit')

    output['data']['query']['address'] = address
    output['data']['metadata']['page'] = page
    output['data']['metadata']['limit'] = size

    pool: pd.DataFrame = \
        tornado_pools[tornado_pools.address == address].iloc[0]

    deposit_txs: Set[str] = get_equal_user_deposit_txs(address)
    deposit_txs_list: List[str] = list(deposit_txs)
    num_deposits: int = len(deposit_txs)

    exact_match_reveals: Set[str] = find_reveals(deposit_txs_list, ExactMatch)
    gas_price_reveals: Set[str] = find_reveals(deposit_txs_list, GasPrice)
    multi_denom_reveals: Set[str] = find_reveals(deposit_txs_list, MultiDenom)

    num_exact_match_reveals: int = len(exact_match_reveals.intersection(deposit_txs))
    num_gas_price_reveals: int = len(gas_price_reveals.intersection(deposit_txs))
    num_multi_denom_reveals: int = len(multi_denom_reveals.intersection(deposit_txs))

    num_compromised: int = \
        num_exact_match_reveals + num_gas_price_reveals + num_multi_denom_reveals

    amount, currency = pool.tags.strip().split()
    stats: Dict[str, Any] = {
        'num_deposits': num_deposits,
        'num_compromised': {
            'all_reveals': num_compromised,
            'exact_match': num_exact_match_reveals,
            'gas_price': num_gas_price_reveals,
            'multi_denom': num_multi_denom_reveals,
        },
        'num_uncompromised': num_deposits - num_compromised
    }

    output['data']['query']['metadata']['amount'] = int(amount)
    output['data']['query']['metadata']['currency'] = currency
    output['data']['query']['metadata']['stats'] = stats
    output['data']['metadata']['compromised_size'] = num_compromised

    output['success'] = 1

    response: str = json.dumps(output)
    rds.set(request_repr, bz2.compress(response.encode('utf-8')))
    return Response(response=response)
