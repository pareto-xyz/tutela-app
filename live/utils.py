import os
import csv
import copy
import heapq
import shutil
import logging
import tempfile
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from os.path import join, dirname, realpath

from src.utils.bigquery import EthereumBigQuery
from src.utils.storage import EthereumStorage

LIVE_DIR: str = realpath(dirname(__file__))
ROOT_DIR: str = realpath(join(LIVE_DIR, '..'))
DATA_DIR: str = realpath(join(ROOT_DIR, 'data'))
STATIC_DIR: str = realpath(join(DATA_DIR, 'static'))
LOG_DIR: str = realpath(join(ROOT_DIR, 'logs'))
WEBAPP_DIR: str = realpath(join(ROOT_DIR, 'webapp'))
WEBAPP_DATA_DIR: str = realpath(join(WEBAPP_DIR, 'static/data'))

CONSTANTS = {
    'live_path': LIVE_DIR,
    'root_path': ROOT_DIR,
    'data_path': DATA_DIR,
    'static_path': STATIC_DIR,
    'log_path': LOG_DIR,
    'webapp_path': WEBAPP_DIR,
    'webapp_data_path': WEBAPP_DATA_DIR,
    'bigquery_project': 'lexical-theory-329617',
    'postgres_db': 'tornado',
    'postgres_user': 'postgres',
}

csv.field_size_limit(2**30)


def export_bigquery_table_to_cloud_bucket(
    src_dataset: str,
    src_table: str,
    dest_bucket: str,
) -> bool:
    handler: EthereumBigQuery = EthereumBigQuery(src_dataset)
    try:
        handler.export_to_bucket(src_table, dest_bucket)
        return True
    except Exception as e:
        print(e.message, e.args)
        return False


def delete_bucket_contents(bucket: str) -> bool:
    handler: EthereumStorage = EthereumStorage()
    try:
        handler.empty_bucket(bucket)
        return True
    except Exception as e:
        print(e.message, e.args)
        return False


def export_cloud_bucket_to_csv(
    bucket: str, out_dir: str) -> Tuple[bool, List[str]]:
    handler: EthereumStorage = EthereumStorage()
    try:
        files: List[str] = handler.export_to_csv(bucket, out_dir)
        return True, files
    except Exception as e:
        print(e.message, e.args)
        return False, []


def execute_bash(cmd: str) -> bool:
    code: int = subprocess.call(cmd, shell=True)
    return code == 0


def get_logger(log_file: str) -> logging.basicConfig:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)

    return logger


def load_data_from_chunks(
    files: List[str], sort_column = 'block_number') -> pd.DataFrame:
    """
    Read dataframes from files and sort by block number.
    """
    chunks: List[pd.DataFrame] = []
    for file_ in files:
        chunk: pd.DataFrame = pd.read_csv(file_)
        chunks.append(chunk)

    data: pd.DataFrame = pd.concat(chunks)
    data.reset_index(inplace=True)
    data.sort_values(sort_column, inplace=True)

    return data


def load_data_from_chunks_low_memory(
    files: List[str], 
    outfile: str,
    sort_column = 'block_number') -> pd.DataFrame:
    """
    Runs external merge sort on a potentially large file.
    """

    print('memory sorting')
    pbar = tqdm(total=len(files))
    for file_ in files:
        # sort the file in place
        memorysort(file_, file_, colname=sort_column)
        pbar.update()
    pbar.close()

    header: List[str] = get_header(files[0])
    merge_idx: int = header.index(sort_column)

    print('running merge sort')
    temp_filename: str = mergesort(files, nway=2, merge_idx=merge_idx)

    shutil.move(temp_filename, outfile)

# -- external sort utilities -- 

def memorysort(
    filename: str,
    outname: str,
    colname: str = 'block_number',
):
    """Sort this CSV file in memory on the given columns"""
    df: pd.DataFrame = pd.read_csv(filename)
    df: pd.DataFrame = df.sort_values(colname)
    df.to_csv(outname, index=False)  # overwriting
    del df  # wipe from memory


def mergesort(sorted_filenames, nway=2, merge_idx=5):
    """Merge two sorted CSV files into a single output."""

    orig_filenames: List[str] = copy.deepcopy(sorted_filenames)
    merge_n: int = 0

    while len(sorted_filenames) > 1:
        # merge_filenames = current files to sort
        # sorted_filenames = remaining files to sort
        merge_filenames, sorted_filenames = \
            sorted_filenames[:nway], sorted_filenames[nway:]
        num_remaining = len(sorted_filenames)
        num_total = len(merge_filenames) + len(sorted_filenames)

        if merge_n % 10 == 0:
            print(f'{merge_n} merged | {num_remaining} remaining | {num_total} total')

        with tempfile.NamedTemporaryFile(delete=False, mode='w') as fp:
            writer: csv.writer = csv.writer(fp)
            merge_n += 1  # increment
            iterators: List[Any]= [
                make_iterator(filename) for filename in merge_filenames]
            # `block_number` is the 4th column
            for row in heapq.merge(*iterators, key=lambda x: int(x[merge_idx])):
                writer.writerow(row)

            sorted_filenames.append(fp.name)

        # these are files to get rid of (don't remove original files)
        extra_filenames: List[str] = list(
            set(merge_filenames) - set(orig_filenames))

        for filename in extra_filenames:
            os.remove(filename)

    final_filename: str = sorted_filenames[0]
    return final_filename


def make_iterator(filename):
    with open(filename, newline='') as fp:
        count = 0
        for row in csv.reader(fp):
            if count == 0:    
                count = 1  # just make it not 0
                continue  # skip header
            yield row


def get_header(filename) -> List[str]:
    with open(filename, newline='') as fp:
        reader = csv.reader(fp, delimiter=',')
        header: List[str] = next(reader)

    return header
