from ens import ENS
from web3 import Web3
from typing import Dict, Any, Optional


def get_balance(address: str, w3: Web3) -> float:
    # returns balance in ETH
    return w3.eth.get_balance(address) / 10**18


def get_ens_name(address: str, ns: ENS) -> str:
    return ns.name(address)


def query_web3(address: str, w3: Web3, ns: ENS) -> Dict[str, Any]:
    address: str = Web3.toChecksumAddress(address)
    return dict(
        balance=get_balance(address, w3),
        ens_name=get_ens_name(address, ns),
    )


def get_ens_address(name: str, ns: ENS) -> Optional[str]:
    return ns.address(name)


def resolve_address(input_: str, ns: ENS) -> str:
    address: str = input_
    if '.eth' in input_:  # only waste compute it .ens is in it
        address: Optional[str] = get_ens_address(input_, ns)
        if address is None:
            address: str = input_  # punt
    return address
