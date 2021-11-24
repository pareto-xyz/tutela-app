import json
import jsonlines

from enum import Enum

class Entity(Enum):
    EOA = 0
    DEPOSIT = 1
    EXCHANGE = 2
    DEX = 3
    DEFI = 4
    ICO_WALLETS = 5
    MINING = 6


class JSONSetEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def to_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp, cls=JSONSetEncoder)


def from_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def to_jsonlines(obj_list, path):
    with jsonlines.open(path, mode='w') as writer:
        writer.write_all(obj_list)


def from_jsonlines(path):
    with jsonlines.open(path) as reader:
        for row in reader:
            yield row
