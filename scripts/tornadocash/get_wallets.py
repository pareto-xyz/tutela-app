"""
Find wallet addresses from transaction addresses.
"""

import re
import time
import urllib
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup

from typing import Optional, Any, Dict, Set, List
from bs4.element import Tag, ResultSet

from src.utils.utils import to_json


def main(args: Any):

    def process_tx(tx: str):
        page: Any = get_etherscan_page(tx)
        data: Dict[str, Any] = get_etherscan_data(page)
        return data['from']['addr']

    clusters: List[Set[str]] = []
    pbar = tqdm(total=get_length(args.csv_path))

    for deposit, withdraw in load_data(args.csv_path):
        deposit_wallet: str = process_tx(deposit)
        time.sleep(0.5)
        withdraw_wallet: str = process_tx(withdraw)
        time.sleep(0.5)

        cluster: Set[str] = {deposit_wallet, withdraw_wallet}
        clusters.append(cluster)

        pbar.update()
    pbar.close()

    to_json(clusters, args.out_json)


def get_length(csv_path: str):
    return len(pd.read_csv(csv_path))


def load_data(csv_path: str):
    df: pd.DataFrame = pd.read_csv(csv_path)

    for row in df.itertuples():
        deposit: str = row.deposit_tx
        withdraw: str = row.withdrawl_tx

        yield deposit, withdraw


def get_etherscan_page(tx_hash: str) -> Optional[bytes]:
    # simulate mozilla browser to download webpage
    user_agent: str = \
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent} 

    uri: str = f'https://etherscan.io/tx/{tx_hash}'
    request = urllib.request.Request(uri, None, headers)  # headers are important!
    response = urllib.request.urlopen(request)

    if response.code == 200:
        data = response.read()
    else:
        data = None

    return data


def get_etherscan_data(page_bytes: Optional[bytes]) -> Optional[str]:
    """
    Pull out all the data from Etherscan on what the transaction is doing.
    """
    if page_bytes is None:  # garbage in, garbage out
        return None

    soup: BeautifulSoup = BeautifulSoup(page_bytes, 'html.parser')
    table_div: Tag = soup.find('div', {'id': 'myTabContent'})
    table_div: Tag = table_div.find('div', {'id': 'ContentPlaceHolder1_maintable'})
    children: ResultSet = table_div.findChildren('div', recursive=False)

    data: Dict[str, Any] = dict()

    for child_div in children:
        messy_text: str = child_div.text

        if 'From:' in messy_text:
            components: ResultSet = child_div.findChildren('div', recursive=False)
            assert len(components) == 2
            raw_text: str = components[1].text.strip()
            type_: str = 'contract' if 'Contract' in raw_text else 'address'
            addr: str = components[1].find('span', {'id': 'spanFromAdd'}).text
            match: Any = re.search(r'\((.*?)\)', raw_text) 
            wallet: Optional[str] = match.group(1) if match is not None else ''
            data['from'] = dict(addr=addr, type=type_, wallet=wallet)
        elif 'To:' in messy_text:
            components: ResultSet = child_div.findChildren('div', recursive=False)
            assert len(components) == 2
            raw_text: str = components[1].text.strip()
            type_: str = 'contract' if 'Contract' in raw_text else 'address'
            addr: str = components[1].find('span', {'id': 'spanToAdd'}).text
            match: Any = re.search(r'\((.*?)\)', raw_text)
            wallet: Optional[str] = match.group(1) if match is not None else ''
            data['to'] = dict(addr=addr, type=type_, wallet=wallet)

    return data



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('csv_path', type=str, help='path to CSV')
    parser.add_argument('out_json', type=str, help='where to save CSV')
    args: Any = parser.parse_args()

    main(args)

