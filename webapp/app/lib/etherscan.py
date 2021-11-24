"""
Currently, I am facing issues with 403 Permission Issues
only on EC2. 
"""
import urllib
from bs4 import BeautifulSoup

from typing import Optional, Any, Dict, List
from bs4.element import Tag, ResultSet, NavigableString


def clean_text(text):
    return text.strip().replace('  ', ' ')


def safe_get_props(component: Tag) -> List[str]:
    """Pops child conent into a stack. Ignores empty."""
    prop_value: List[str] = []
    for obj in component.children:
        if isinstance(obj, NavigableString):
            if len(obj.strip()) == 0 :
                continue
            else:
                prop_value.append(obj.strip())
        else:
            if len(obj.text) > 0:
                prop_value.append(obj.text)
    return prop_value


def get_etherscan_page(address: str) -> Optional[bytes]:
    # https://stackoverflow.com/questions/28396036/python-3-4-urllib-request-error-http-403
    # simulate mozilla browser to download webpage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    uri: str = f'https://etherscan.io/address/{address}'
    request = urllib.request.Request(uri, headers = headers)
    response = urllib.request.urlopen(request)

    if response.code == 200:
        data = response.read()
    else:
        data = None

    return data


def get_etherscan_data(page_bytes: Optional[bytes]) -> Dict[str, Any]:
    """
    Pull out all the data from Etherscan on what the transaction is doing.
    This funciton is riddled with safe returns.
    """
    if page_bytes is None:  # garbage in, garbage out
        return None

    def uncapitalize(s):
        if len(s) == 0:
            return s
        elif len(s) == 1:
            return s.lower()
        else:
            if s[0] == s[0].upper():
                s = s[0].lower() + s[1:]
            return s

    soup: BeautifulSoup = BeautifulSoup(page_bytes, 'html.parser')
    summary_div: Tag = soup.find('div', {'id': 'ContentPlaceHolder1_divSummary'})
    if summary_div is None: return {}  # nothing to do

    summary_res: ResultSet = summary_div.findChildren('div', recursive=False)
    if len(summary_res) == 0: return {}  # nothing to do

    summary: Tag = summary_res[0]

    body_res: ResultSet = summary.findChildren('div', {'class': 'card-body'})
    if len(body_res) == 0: return {}  # nothing to do
    body: Tag = body_res[0]

    columns: ResultSet = body.findChildren('div', recursive=False)
    if len(columns) < 2: return {}  # incorrect number of columns.

    data: Dict[str, Any] = dict()  # collect all data

    # handle the first two divs the same way
    for column in columns[:2]:
        items: ResultSet = column.findChildren('div', recursive=False)
        if len(items) != 2:
            continue  # this is not unexpected so we should ignore
        k, v = items[0].text.strip(), items[1].text.strip()
        k: str = k.replace(':', '').strip()
        k: str = uncapitalize(k)
        data[k] = v

    if len(columns) > 2:
        # deal with third column
        content_res: ResultSet = columns[2].findChildren(
            'a', {'id': 'availableBalanceDropdown'})
        if len(content_res) > 0:
            content: Tag = content_res[0]
            token_value: str = content.text.strip()
            token_value: str = token_value.split('\n')[0]
            data['tokenValue'] = token_value

            token_res: ResultSet = content.findChildren('span', recursive=False)
            if len(token_res) > 0:
                token: Tag = token_res[0]
                num_tokens: int = int(token.text)
                data['numTokenContracts'] = num_tokens

    return data


def query_etherscan(address: str):
    data = get_etherscan_page(address)
    return get_etherscan_data(data)


if __name__ == "__main__":
    from pprint import pprint
    address: str = '0x49516e20b5692839f877a7e9aa62006a5d02a7b1'
    data = get_etherscan_page(address)
    data = get_etherscan_data(data)
    pprint(data)
