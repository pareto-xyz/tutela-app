import logging
import subprocess
from os.path import join, dirname, realpath

LIVE_DIR: str = realpath(join(dirname(__file__), '..'))
ROOT_DIR: str = realpath(join(LIVE_DIR, '..'))
DATA_DIR: str = realpath(join(ROOT_DIR, 'data'))
LOG_DIR: str = realpath(join(ROOT_DIR, 'logs'))

CONSTANTS = {
    'live_path': LIVE_DIR,
    'root_path': ROOT_DIR,
    'data_path': DATA_DIR,
    'log_path': LOG_DIR,
    'bigquery_project': 'lexical-theory-329617',
}


def export_bigquery_table_to_cloud_bucket():
    pass


def export_cloud_bucket_to_csv():
    pass


def execute_bash(cmd: str) -> bool:
    code: int = subprocess.call(cmd, shell=True, executable='/bin/bash')
    return code == 0


def get_logger(log_file: str) -> logging.basicConfig:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(worker)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)
