import bz2
import math
import json
import numpy as np
from typing import Dict, Optional, List, Any, Set

from app import app, w3, ns, known_addresses, rds
from app.models import \
    Address, ExactMatch, GasPrice, \
    TornadoDeposit, TornadoWithdraw
from app.utils import \
    get_anonymity_score, get_order_command, \
    entity_to_int, entity_to_str, to_dict, \
    RequestChecker, default_response, \
    NAME_COL, ENTITY_COL, CONF_COL, EOA, DEPOSIT, EXCHANGE
from app.lib.w3 import query_web3

from flask import request, Response
from flask import render_template
from sqlalchemy import or_

from app.utils import get_known_attrs

PAGE_LIMIT = 50
HARD_MAX: int = 1000


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['GET'])
def cluster():
    return render_template('cluster.html')


@app.route('/search', methods=['GET'])
def search():
    table_cols: Set[str] = set(Address.__table__.columns.keys())
    checker: RequestChecker = RequestChecker(
        request,
        table_cols,
        entity_key = ENTITY_COL,
        conf_key = CONF_COL,
        name_key = NAME_COL,
        default_page = 0,
        default_limit = PAGE_LIMIT,
    )
    is_valid_request: bool = checker.check()
    output: Dict[str, Any] = default_response()

    if not is_valid_request: 
        return Response(output)

    address: str = checker.get('address')
    page: int = checker.get('page')
    size: int = checker.get('limit')
    sort_by: str = checker.get('sort_by')
    desc_sort: str = checker.get('desc_sort')
    filter_by: List[Any] = checker.get('filter_by')

    if rds.exists(address):  # check if this exists in our cache
        response: str = bz2.decompress(rds.get(address)).decode('utf-8')
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

    def query_exact_match_heuristic(address: str) -> Set[str]:
        """
        Given an address, find out how many times this address' txs
        appear in a exact match heuristic. Returns a list of all 
        transactions with the same cluster as this address.
        """
        rows: Optional[List[ExactMatch]] = \
            ExactMatch.query.filter_by(address = address).all()

        cluster_txs: List[str] = []

        if (len(rows) > 0):
            for row in rows:
                # find cluster for this transaction (w/ this address)
                cluster: List[ExactMatch] = ExactMatch.query.filter_by(
                    cluster = row.cluster).all()
                cluster: List[str] = [member.transaction for member in cluster]
                cluster_txs.extend(cluster)

        return set(cluster_txs)  # no duplicates

    def query_gas_price_heuristic(address: str) -> Set[str]:
        """
        Given an address, find out how many times this address' txs 
        appears in a same gas price reveal. We will return the tx info too?
        """
        rows: Optional[List[GasPrice]] = \
            GasPrice.query.filter_by(address = address).all()

        cluster_txs: List[str] = []

        if len(rows) > 0:
            for row in rows:
                # find cluster for this tx
                cluster: List[GasPrice] = GasPrice.query.filter_by(
                    cluster = row.cluster).all()
                cluster: List[str] = [member.transaction for member in cluster]
                cluster_txs.extend(cluster)

        return set(cluster_txs)  # no duplicates

    def query_deposit_reuse_heuristic(
        address: str, limit: int = HARD_MAX) -> Set[str]:
        """
        For the given address, find all other EOA addresses in the DAR cluster.
        The purpose of this is when we are computing # of deposit addr, we 
        compute using all of the addresses in cluster, rather than just the
        current address.
        """
        cluster: Set[str] = {address}

        addr: Address = Address.query.filter_by(address = address).first()
        if addr is not None:
            cluster: List[Address] = Address.query.filter_by(
                user_cluster = addr.user_cluster, 
                entity = entity_to_int(EOA),
            ).limit(limit).all()
            cluster: Set[str] = set([c.address for c in cluster])

        return cluster

    def query_address_tornado_stats(address: str) -> Dict[str, int]:
        """
        Given a user address, we want to supply a few statistics:

        1) Number of deposits made to Tornado pools.
        2) Number of withdraws made to Tornado pools.
        3) Number of deposits made that are part of a cluster or of a TCash reveal.
        """
        exact_match_txs: Set[str] = query_exact_match_heuristic(address)
        gas_price_txs: Set[str] = query_gas_price_heuristic(address)
        reveal_txs: Set[str] = exact_match_txs.union(gas_price_txs)

        # find all txs where the from_address is the current user.
        deposits: Optional[List[TornadoDeposit]] = \
            TornadoDeposit.query.filter_by(from_address = address).all()
        deposit_txs: Set[str] = set([d.hash for d in deposits])
        num_deposit: int = len(deposit_txs)

        # find all txs where the recipient_address is the current user
        withdraws: Optional[List[TornadoWithdraw]] = \
            TornadoWithdraw.query.filter_by(recipient_address = address).all()
        num_withdraw: int = len(set([w.hash for w in withdraws]))

        # compute number of txs compromised by TCash heuristics
        num_remain: int = len(deposit_txs - reveal_txs)
        num_compromised: int = num_deposit - num_remain

        stats: Dict[str, int] = dict(
            address_num_deposit = num_deposit,
            address_num_withdraw = num_withdraw,
            address_num_compromised = num_compromised,
        )
        return stats

    def query_cluster_tornado_stats(address: str) -> Dict[str, int]:
        """
        Same as `query_address_tornado_stats but for a cluster of
        addresses obtained from deposit address reuse.
        """
        # find EOA addresses in the same cluster as address
        cluster: Set[str] = query_deposit_reuse_heuristic(address)
        cluster_stats: Dict[str, int] = dict(
            cluster_num_deposit = 0,
            cluster_num_withdraw = 0,
            cluster_num_compromised = 0,
        )
        reveal_txs: Set[str] = set()
        deposit_txs: Set[str] = set()
        num_deposit: int = 0
        num_withdraw: int = 0

        for address in cluster:
            exact_match_txs: Set[str] = query_exact_match_heuristic(address)
            gas_price_txs: Set[str] = query_gas_price_heuristic(address)
            cur_reveal_txs: Set[str] = exact_match_txs.union(gas_price_txs)

            deposits: Optional[List[TornadoDeposit]] = \
                TornadoDeposit.query.filter_by(from_address = address).all()
            cur_deposit_txs: Set[str] = set([d.hash for d in deposits])
            cur_num_deposit: int = len(cur_deposit_txs)

            withdraws: Optional[List[TornadoWithdraw]] = \
                TornadoWithdraw.query.filter_by(recipient_address = address).all()
            cur_num_withdraw: int = len(set([w.hash for w in withdraws]))

            # combine with other addresses
            reveal_txs: Set[str] = reveal_txs.union(cur_reveal_txs)
            deposit_txs: Set[str] = deposit_txs.union(cur_deposit_txs)
            num_deposit += cur_num_deposit
            num_withdraw += cur_num_withdraw

        num_remain: int = len(deposit_txs - reveal_txs)
        num_compromised: int = num_deposit - num_remain

        stats: Dict[str, int] = dict(
            cluster_num_deposit = num_deposit,
            cluster_num_withdraw = num_withdraw,
            cluster_num_compromised = num_compromised,
        )
        return stats


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
                **to_dict(addr, table_cols, to_transform=[('entity', entity_to_str)])
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
        address_tornado_dict: Dict[str, Any] = query_address_tornado_stats(address)
        output['data']['tornado']['summary'].update(address_tornado_dict)

        if addr.entity == entity_to_int(EOA):
            # only add cluster information if current address is an EOA.
            cluster_tornado_dict: Dict[str, Any] = query_cluster_tornado_stats(address)
            output['data']['tornado']['summary'].update(cluster_tornado_dict)

        # if `addr` doesnt exist, then we assume no clustering
        output['success'] = 1

    response: str = json.dumps(output)
    rds.set(address, bz2.compress(response.encode('utf-8')))  # add to cache

    return Response(response=response)
