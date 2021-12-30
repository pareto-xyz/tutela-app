import bz2
import math
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta 
from typing import Dict, Optional, List, Any, Set

from app import app, w3, ns, rds, known_addresses, tornado_pools
from app.models import \
    Address, ExactMatch, GasPrice, MultiDenom, LinkedTransaction, TornMining, \
    TornadoDeposit, TornadoWithdraw, Embedding, DepositTransaction
from app.utils import \
    get_anonymity_score, get_order_command, \
    entity_to_int, entity_to_str, to_dict, \
    heuristic_to_str, is_valid_address, \
    is_tornado_address, get_equal_user_deposit_txs, find_reveals, \
    AddressRequestChecker, TornadoPoolRequestChecker, TransactionRequestChecker, \
    default_address_response, default_tornado_response, default_transaction_response, \
    NAME_COL, ENTITY_COL, CONF_COL, EOA, DEPOSIT, EXCHANGE, NODE, \
    GAS_PRICE_HEUR, DEPO_REUSE_HEUR, DIFF2VEC_HEUR, SAME_NUM_TX_HEUR, \
    SAME_ADDR_HEUR, LINKED_TX_HEUR, TORN_MINE_HEUR, DIFF2VEC_HEUR
from app.lib.w3 import query_web3, get_ens_name, resolve_address

from flask import request, Request, Response
from flask import render_template
from sqlalchemy import or_

from app.utils import get_known_attrs, get_display_aliases
from app.utils import heuristic_to_int

PAGE_LIMIT = 50
HARD_MAX: int = 1000


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
@app.route('/cluster', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/utils/aliases', methods=['GET'])
def alias():
    response: str = json.dumps(get_display_aliases())
    return Response(response=response)


@app.route('/utils/istornado', methods=['GET'])
def istornado():
    address: str = request.args.get('address', '')
    address: str = resolve_address(address, ns)
    address: str = address.lower()

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
        return Response(json.dumps(output))

    is_tornado: bool = int(is_tornado_address(address))
    if not is_tornado:
        amount = None 
        currency = None
    else:
        pool: pd.DataFrame = \
            tornado_pools[tornado_pools.address == address].iloc[0]
        amount, currency = pool.tags.strip().split()
        amount = int(amount)

    output['data']['is_tornado'] = is_tornado
    output['data']['amount'] = amount
    output['data']['currency'] = currency
    output['success'] = 1

    response: str = json.dumps(output)
    return Response(response)


@app.route('/utils/gettornadopools', methods=['GET'])
def get_tornado_pools():
    pools = []
    for _, pool in tornado_pools.iterrows():
        # amount, currency = pool.tags.strip().split()
        pools.append({
            'address': pool.address, 
            'name': pool.tags,
        })

    output: Dict[str, Any] = {
        'data': {'pools': pools},
        'success': 1,
    }
    response: str = json.dumps(output) 
    return Response(response)


@app.route('/search', methods=['GET'])
def search():
    address: str = request.args.get('address', '')
    # after this call, we should expect address to be an address
    address: str = resolve_address(address, ns)
    address: str = address.lower()

    # do a simple check that the address is valid
    if not is_valid_address(address):
        return default_address_response()

    # check if address is a tornado pool or not
    is_tornado: bool = is_tornado_address(address)

    # change request object
    request.args = dict(request.args)
    request.args['address'] = address
    
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

    # search for these txs in the reveal tables
    exact_match_reveals: Set[str] = find_reveals(deposit_txs, ExactMatch)
    gas_price_reveals: Set[str] = find_reveals(deposit_txs, GasPrice)
    multi_denom_reveals: Set[str] = find_reveals(deposit_txs, MultiDenom)
    linked_tx_reveals: Set[str] = find_reveals(deposit_txs, LinkedTransaction)
    torn_mine_reveals: Set[str] = find_reveals(deposit_txs, TornMining)

    def format_compromised(
        exact_match_reveals: Set[str],
        gas_price_reveals: Set[str],
        multi_denom_reveals: Set[str],
        linked_tx_reveals: Set[str],
        torn_mine_reveals: Set[str],
    ) -> List[Dict[str, Any]]:
        compromised: List[Dict[str, Any]] = []
        for reveal in exact_match_reveals:
            compromised.append({'heuristic': heuristic_to_str(1), 'transaction': reveal})
        for reveal in gas_price_reveals:
            compromised.append({'heuristic': heuristic_to_str(2), 'transaction': reveal})
        for reveal in multi_denom_reveals:
            compromised.append({'heuristic': heuristic_to_str(3), 'transaction': reveal})
        for reveal in linked_tx_reveals:
            compromised.append({'heuristic': heuristic_to_str(4), 'transaction': reveal})
        for reveal in torn_mine_reveals:
            compromised.append({'heuristic': heuristic_to_str(5), 'transaction': reveal})
        return compromised

    # add compromised sets to response
    compromised: List[Dict[str, Any]] = format_compromised(
        exact_match_reveals, gas_price_reveals, multi_denom_reveals, 
        linked_tx_reveals, torn_mine_reveals)
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

    address: str = checker.get('address').lower()
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
        addr: Optional[Address],
        ens_name: Optional[str] = None,
        exchange_weight: float = 0.1,
        slope: float = 0.1,
        extra_cluster_sizes: List[int] = [],
        extra_cluster_confs: List[float] = []
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

        If ens_name is provided and not empty, we cap the anonymity score at 90.
        If addr is None, we assume clusters are specified in extra_cluster_*.
        """
        cluster_confs: List[float] = extra_cluster_sizes
        cluster_sizes: List[float] = extra_cluster_confs

        if addr is not None:
            if addr.entity == entity_to_int(DEPOSIT):
                return -1  # represents N/A
            elif addr.entity == entity_to_int(EXCHANGE):
                return 0   # CEX have no anonymity

            assert addr.entity == entity_to_int(EOA), \
                f'Unknown entity: {entity_to_str(addr.entity)}'

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

        if ens_name is not None:
            if len(ens_name) > 0 and '.eth' in ens_name:
                # having an ENS name caps your maximum anonymity score
                score: float = min(score, 0.90)

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

    def query_tornado_stats(address: str) -> Dict[str, Any]:
        """
        Given a user address, we want to supply a few statistics:

        1) Number of deposits made to Tornado pools.
        2) Number of withdraws made to Tornado pools.
        3) Number of deposits made that are part of a cluster or of a TCash reveal.
        """
        exact_match_txs: Set[str] = query_heuristic(address, ExactMatch)
        gas_price_txs: Set[str] = query_heuristic(address, GasPrice)
        multi_denom_txs: Set[str] = query_heuristic(address, MultiDenom)
        linked_txs: Set[str] = query_heuristic(address, LinkedTransaction)
        torn_mine_txs: Set[str] = query_heuristic(address, TornMining)

        reveal_txs: Set[str] = set().union(
            exact_match_txs, gas_price_txs, multi_denom_txs, 
            linked_txs, torn_mine_txs)

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
        num_remain_linked_tx: int = len(all_txs - linked_txs)
        num_remain_torn_mine: int = len(all_txs - torn_mine_txs)

        num_compromised: int = num_all - num_remain
        num_compromised_exact_match = num_all - num_remain_exact_match
        num_compromised_gas_price = num_all - num_remain_gas_price
        num_compromised_multi_denom = num_all - num_remain_multi_denom
        num_compromised_linked_tx = num_all - num_remain_linked_tx
        num_compromised_torn_mine = num_all - num_remain_torn_mine

        # compute number of txs compromised by TCash heuristics
        stats: Dict[str, Any] = dict(
            num_deposit = num_deposit,
            num_withdraw = num_withdraw,
            num_compromised = dict(
                all_reveals = num_compromised,
                num_compromised_exact_match = num_compromised_exact_match,
                num_compromised_gas_price = num_compromised_gas_price,
                num_compromised_multi_denom = num_compromised_multi_denom,
                num_compromised_linked_tx = num_compromised_linked_tx,
                num_compromised_torn_mine = num_compromised_torn_mine,
                hovers = dict(
                    num_compromised_exact_match = '# of deposits to/withdrawals from tornado cash pools linked through the address match heuristic. Address match links transactions if a unique address deposits and withdraws to a Tornado Cash pool.',
                    num_compromised_gas_price = '# of deposits to/withdrawals from tornado cash pools linked through the unique gas price heuristic. Unique gas price links deposit and withdrawal transactions that use a unique and specific (e.g. 3.1415) gas price.',
                    num_compromised_multi_denom = '# of deposit/withdrawals into tornado cash pools linked through the multi-denomination reveal. Multi-denomination reveal is when a “source” wallet mixes a specific set of denominations and your “destination” wallet withdraws them all. For example, if you mix 3x 10 ETH, 2x 1 ETH, 1x 0.1 ETH to get 32.1 ETH, you could reveal yourself within the Tornado protocol if no other wallet has mixed this exact denomination set.',
                    num_compromised_linked_tx = '# of deposits to/withdrawals from tornado cash pools linked through the linked address reveal. Linked address reveal connects wallets that interact outside of Tornado Cash.',
                    num_compromised_torn_mine = '# of deposits to/withdrawals from tornado cash pools linked through the TORN mining reveal. Careless swapping of Anonymity Points to TORN tokens reveal information of when deposits were made.',
                )
            ),
            num_uncompromised = num_all - num_compromised,
            hovers = dict(
                num_deposit = '# of deposit transactions into tornado cash pools.',
                num_withdraw = '# of withdrawal transactions from tornado cash pools.',
                num_compromised = '# of deposits to/withdrawals from tornado cash pools that may be linked through the mis-use of Tornado cash.',
            )
        )
        return stats

    def query_diff2vec(node: Embedding) -> List[Dict[str, Any]]:
        """
        Search the embedding table to fetch neighbors from Diff2Vec cluster.
        """
        cluster: List[Dict[str, Any]] = []
        cluster_conf: float = 0

        if node is not None:
            neighbors: List[int] = json.loads(node.neighbors)
            distances: List[float] = json.loads(node.distances)

            for neighbor, distance in zip(neighbors, distances):
                # swap terms b/c of upload accident
                neighbor, distance = distance, neighbor
                if neighbor == address: continue  # skip
                member: Dict[str, Any] = {
                    'address': neighbor,
                    # '_distance': distance,
                     # add one to make max 1
                    'conf': round(float(1./abs(10.*distance+1.)), 3),
                    'heuristic': DIFF2VEC_HEUR, 
                    'entity': NODE,
                    'ens_name': get_ens_name(neighbor, ns),
                }
                cluster.append(member)
                cluster_conf += member['conf']

        cluster_size: int = len(cluster)
        cluster_conf: float = cluster_conf / float(cluster_size)

        return cluster, cluster_size, cluster_conf


    if len(address) > 0:
        offset: int = page * size

        # --- check web3 for information ---
        web3_resp: Dict[str, Any] = query_web3(address, w3, ns)
        metadata_: Dict[str, Any] = output['data']['query']['metadata']
        output['data']['query']['metadata'] = {**metadata_, **web3_resp}

        # --- check tornado queries ---
        # Note that this is out of the `Address` existence check
        tornado_dict: Dict[str, Any] = query_tornado_stats(address)
        output['data']['tornado']['summary']['address'].update(tornado_dict)

        # --- search for address in DAR and Dff2Vec tables ---
        addr: Optional[Address] = Address.query.filter_by(address = address).first()
        node: Optional[Embedding] = Embedding.query.filter_by(address = address).first()

        # --- Case #1 : address can be found in the DAR Address table --- 
        if addr is not None: 
            entity: str = entity_to_str(addr.entity)
            if addr.meta_data is None: addr.meta_data = '{}'
            addr_metadata: Dict[str, Any] = json.loads(addr.meta_data)  # load metadata
            if 'ens_name' in addr_metadata: del addr_metadata['ens_name']  # no override
            metadata_: Dict[str, Any] = output['data']['query']['metadata']
            output['data']['query']['metadata'] = {**metadata_, **addr_metadata}

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

            # find Diff2Vec embeddings and add to front of cluster
            diff2vec_cluster, diff2vec_size, diff2vec_conf = query_diff2vec(node)
            cluster: List[Dict[str, Any]] = diff2vec_cluster + cluster
            cluster_size += len(diff2vec_cluster)

            output['data']['cluster'] = cluster
            output['data']['metadata']['cluster_size'] = cluster_size
            output['data']['metadata']['num_pages'] = int(math.ceil(cluster_size / size))

            # --- compute anonymity score using hyperbolic fn ---
            anon_score = compute_anonymity_score(
                addr,
                ens_name = web3_resp['ens_name'],
                # seed computing anonymity score with diff2vec + tcash reveals
                extra_cluster_sizes = [
                    diff2vec_size,
                    tornado_dict['num_compromised']['num_compromised_exact_match'],
                    tornado_dict['num_compromised']['num_compromised_gas_price'],
                    tornado_dict['num_compromised']['num_compromised_multi_denom'],
                    tornado_dict['num_compromised']['num_compromised_linked_tx'],
                    tornado_dict['num_compromised']['num_compromised_torn_mine'],
                ], 
                extra_cluster_confs = [
                    diff2vec_conf,
                    1.,
                    1.,
                    0.5,
                    0.25,
                    0.25,
                ],
            )
            anon_score: float = round(anon_score, 3)  # brevity is a virtue
            output['data']['query']['anonymity_score'] = anon_score

        # --- Case #2: address is not in the DAR Address table but is 
        #              in Embedding (Diff2Vec) table --- 
        elif node is not None:
            # find Diff2Vec embeddings and add to front of cluster
            cluster, cluster_size, conf = query_diff2vec(node)

            anon_score = compute_anonymity_score(
                None,
                ens_name = web3_resp['ens_name'],
                # seed computing anonymity score with diff2vec + tcash reveals
                extra_cluster_sizes = [
                    cluster_size,
                    tornado_dict['num_compromised']['num_compromised_exact_match'],
                    tornado_dict['num_compromised']['num_compromised_gas_price'],
                    tornado_dict['num_compromised']['num_compromised_multi_denom'],
                    tornado_dict['num_compromised']['num_compromised_linked_tx'],
                    tornado_dict['num_compromised']['num_compromised_torn_mine'],
                ], 
                extra_cluster_confs = [
                    conf,
                    1.,
                    1.,
                    0.5,
                    0.25,
                    0.25,
                ],
            )
            anon_score: float = round(anon_score, 3)
            output['data']['query']['anonymity_score'] = anon_score
            output['data']['query']['heuristic'] = DIFF2VEC_HEUR
            output['data']['query']['entity'] = NODE
            output['data']['query']['conf'] = round(conf, 3)
            output['data']['cluster'] = cluster
            output['data']['metadata']['cluster_size'] = cluster_size
            output['data']['metadata']['num_pages'] = int(math.ceil(cluster_size / size))

        # Check if we know existing information about this address 
        known_lookup: Dict[str, Any] = get_known_attrs(known_addresses, address)
        
        if len(known_lookup) > 0:
            query_metadata: Dict[str, Any] = output['data']['query']['metadata']
            output['data']['query']['metadata'] = {**query_metadata, **known_lookup}

            # if you are on the top 20k users list, no anonymity
            output['data']['query']['anonymity_score'] = 0

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

    address: str = checker.get('address').lower()
    page: int = checker.get('page')
    size: int = checker.get('limit')
    return_tx: bool = checker.get('return_tx')

    output['data']['query']['address'] = address
    output['data']['metadata']['page'] = page
    output['data']['metadata']['limit'] = size

    pool: pd.DataFrame = \
        tornado_pools[tornado_pools.address == address].iloc[0]

    deposit_txs: Set[str] = get_equal_user_deposit_txs(address)
    num_deposits: int = len(deposit_txs)

    exact_match_reveals: Set[str] = find_reveals(deposit_txs, ExactMatch)
    gas_price_reveals: Set[str] = find_reveals(deposit_txs, GasPrice)
    multi_denom_reveals: Set[str] = find_reveals(deposit_txs, MultiDenom)
    linked_tx_reveals: Set[str] = find_reveals(deposit_txs, LinkedTransaction)
    torn_mine_reveals: Set[str] = find_reveals(deposit_txs, TornMining)

    reveal_txs: Set[str] = set().union(
        exact_match_reveals, gas_price_reveals, multi_denom_reveals, 
        linked_tx_reveals, torn_mine_reveals)

    num_exact_match_reveals: int = len(exact_match_reveals)
    num_gas_price_reveals: int = len(gas_price_reveals)
    num_multi_denom_reveals: int = len(multi_denom_reveals)
    num_linked_tx_reveals: int = len(linked_tx_reveals)
    num_torn_mine_reveals: int = len(torn_mine_reveals)

    num_compromised: int = len(reveal_txs)
    amount, currency = pool.tags.strip().split()
    stats: Dict[str, Any] = {
        'num_deposits': num_deposits,
        'tcash_num_compromised': {
            'all_reveals': num_compromised,
            'exact_match': num_exact_match_reveals,
            'gas_price': num_gas_price_reveals,
            'multi_denom': num_multi_denom_reveals,
            'linked_tx': num_linked_tx_reveals,
            'torn_mine': num_torn_mine_reveals,
        },
        'tcash_num_uncompromised': num_deposits - num_compromised
    }

    if return_tx:
        output['data']['deposits'] = list(deposit_txs)
        output['data']['compromised'] = {
            'exact_match': list(exact_match_reveals),
            'gas_price': list(gas_price_reveals),
            'multi_denom': list(multi_denom_reveals),
            'linked_tx': list(linked_tx_reveals),
            'torn_mine': list(torn_mine_reveals),
        }

    output['data']['query']['metadata']['amount'] = float(amount)
    output['data']['query']['metadata']['currency'] = currency
    output['data']['query']['metadata']['stats'] = stats
    output['data']['metadata']['compromised_size'] = num_compromised

    output['success'] = 1

    response: str = json.dumps(output)
    rds.set(request_repr, bz2.compress(response.encode('utf-8')))
    return Response(response=response)


@app.route('/transaction', methods=['GET'])
def transaction():
    return render_template('transaction.html')


@app.route('/search/transaction', methods=['GET'])
def search_transaction():
    address: str = request.args.get('address', '')
    address: str = resolve_address(address, ns)
    address: str = address.lower()

    if not is_valid_address(address):
        return default_transaction_response()

    request.args = dict(request.args)
    request.args['address'] = address

    checker: TransactionRequestChecker = TransactionRequestChecker(
        request,
        default_page = 0,
        default_limit = PAGE_LIMIT,
        default_window = '1yr',
    )
    is_valid_request: bool = checker.check()
    output: Dict[str, Any] = default_transaction_response()

    if not is_valid_request:
        return Response(output)

    address: str = checker.get('address').lower()
    page: int = checker.get('page')
    size: int = checker.get('limit')
    window: str = checker.get('window')

    request_repr: str = checker.to_str()

    if rds.exists(request_repr):
        response: str = bz2.decompress(rds.get(request_repr)).decode('utf-8')
        return Response(response=response)

    output['data']['query']['address'] = address
    output['data']['metadata']['page'] = page
    output['data']['metadata']['limit'] = size
    output['data']['metadata']['window'] = window

    # --

    def find_tcash_matches(address: str, Heuristic: Any, identifier: int
    ) -> List[Dict[str, Any]]:

        rows: List[Heuristic] = \
            Heuristic.query.filter(Heuristic.address == address).all()
        rows: List[Dict[str, Any]] = [
            {'transaction': row.transaction, 'block': row.block_number, 
             'timestamp': row.block_ts, 'heuristic': identifier,
             'metadata': {}} for row in rows]
        return rows

    def find_dar_matches(address: str) -> List[Dict[str, Any]]:
        rows: List[DepositTransaction] = \
            DepositTransaction.query.filter(DepositTransaction.address == address).all()
        rows: List[Dict[str, Any]] = [
            {'transaction': row.transaction, 'block': row.block_number, 
             'timestamp': row.block_ts, 'heuristic': DEPO_REUSE_HEUR,
             'metadata': {'deposit': row.deposit}} for row in rows]
        return rows

    dar_matches: List[Dict[str, Any]] = find_dar_matches(address)
    same_addr_matches: List[Dict[str, Any]] = \
        find_tcash_matches(address, ExactMatch, heuristic_to_int(SAME_ADDR_HEUR))
    gas_price_matches: List[Dict[str, Any]] = \
        find_tcash_matches(address, GasPrice, heuristic_to_int(GAS_PRICE_HEUR))
    same_num_tx_matches: List[Dict[str, Any]] = \
        find_tcash_matches(address, MultiDenom, heuristic_to_int(SAME_NUM_TX_HEUR))
    linked_tx_matches: List[Dict[str, Any]] = \
        find_tcash_matches(address, LinkedTransaction, heuristic_to_int(LINKED_TX_HEUR))
    torn_mine_matches: List[Dict[str, Any]] = \
        find_tcash_matches(address, TornMining, heuristic_to_int(TORN_MINE_HEUR))

    transactions: List[Dict[str, Any]] = \
        dar_matches + same_addr_matches + gas_price_matches + same_num_tx_matches + \
        linked_tx_matches + torn_mine_matches

    plotdata: List[Dict[str, Any]] = make_weekly_plot(transactions)

    output['data']['query']['metadata']['stats']['num_transactions'] += len(transactions)
    output['data']['query']['metadata']['stats']['num_ethereum'][DEPO_REUSE_HEUR] += len(dar_matches)
    output['data']['query']['metadata']['stats']['num_tcash'][SAME_ADDR_HEUR] += len(same_addr_matches)
    output['data']['query']['metadata']['stats']['num_tcash'][GAS_PRICE_HEUR] += len(gas_price_matches)
    output['data']['query']['metadata']['stats']['num_tcash'][SAME_NUM_TX_HEUR] += len(same_num_tx_matches)
    output['data']['query']['metadata']['stats']['num_tcash'][LINKED_TX_HEUR] += len(linked_tx_matches)
    output['data']['query']['metadata']['stats']['num_tcash'][TORN_MINE_HEUR] += len(torn_mine_matches)

    # sort by timestamp
    transactions: List[Dict[str, Any]] = sorted(transactions, key = lambda x: x['timestamp'])

    def tx_datetime_to_str(raw_transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transactions: List[Dict[str, Any]] = []
        for tx in raw_transactions:
            tx['timestamp'] = tx['timestamp'].strftime('%m/%d/%Y')
            transactions.append(tx)
        return transactions

    # remove datetime objects
    transactions: List[Dict[str, Any]] = tx_datetime_to_str(transactions)

    output['data']['transactions'] = transactions
    output['data']['plotdata'] = plotdata
    output['success'] = 1

    response: str = json.dumps(output)
    rds.set(request_repr, bz2.compress(response.encode('utf-8')))  # add to cache

    return Response(response=response)


def make_weekly_plot(
    transactions: List[Dict[str, Any]], window = '1yr') -> List[Dict[str, Any]]:
    """
    Make the data grouped by heuristics by week.

    @window: [6mth, 1yr, 5yr]
    """
    assert window in ['6mth', '1yr', '5yr'], f'Invalid window: {window}.'
    today: datetime = datetime.today()
    
    if window == '6mth':
        delta: relativedelta = relativedelta(months=6)
    elif window == '1yr':
        delta: relativedelta = relativedelta(months=12)
    elif window == '5yr':
        delta: relativedelta = relativedelta(months=12*5)
    else:
        raise Exception(f'Window {window} not supported.')

    start: datetime = today - delta
    data: List[Dict[str, Any]] = []
    cur_start: datetime = copy.copy(start)
    cur_end: datetime = cur_start + relativedelta(weeks=1)
    count: int = 0

    while cur_end <= today:
        counts: Dict[str, int] = {
            DEPO_REUSE_HEUR: 0,
            SAME_ADDR_HEUR: 0,
            GAS_PRICE_HEUR: 0,
            SAME_NUM_TX_HEUR: 0,
            LINKED_TX_HEUR: 0,
            TORN_MINE_HEUR: 0,
        }
        for transaction in transactions:
            ts: datetime = transaction['timestamp']
            if (ts >= cur_start) and (ts <= cur_end):
                counts[transaction['heuristic']] += 1

        start_date: str = cur_start.strftime('%m/%d/%Y')
        end_date: str = cur_end.strftime('%m/%d/%Y')

        row: Dict[str, Any] = {
            'index': count,
            'start_date': start_date,
            'end_date': end_date,
            'name': f'{start_date}-{end_date}',
            **counts,
        }
        data.append(row)

        cur_start: datetime = copy.copy(cur_end)
        cur_end: datetime = cur_start + relativedelta(weeks=1)
        count += 1

    return data

