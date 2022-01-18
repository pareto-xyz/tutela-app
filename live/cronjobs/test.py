import os
from datetime import datetime
from os.path import join, isfile
from live import utils


def main():
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'test.log')
    if isfile(log_file):
        os.remove(log_file)

    with open(log_file, 'w') as fp:
        now: datetime = datetime.now()
        fp.write('hello world: ' + now.strftime("%d/%m/%Y %H:%M:%S") + '\n')


if __name__ == "__main__":
    main()
