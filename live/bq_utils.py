from typing import List
from live import utils


def make_bq_delete(table: str, flags: List[str]) -> str:
    project: str = utils.CONSTANTS['bigquery_project']
    flags: str = ' '.join(flags)
    statement: str = f'delete from {project}.{table} where true'
    query: str = f"bq query {flags} '{statement}'"
    return query


def make_bq_query(
    select: str, where_clauses: List[str] = [], flags: List[str] = []) -> str:
    flags: str = ' '.join(flags)
    where_clauses: List[str] = [f'({clause})' for clause in where_clauses]
    where_clauses: str = ' and '.join(where_clauses)
    query: str = f"bq query {flags} '{select} where {where_clauses}'"
    return query


def make_bq_load(table: str, csv_path: str, schema: str) -> str:
    project: str = utils.CONSTANTS['bigquery_project']
    flags: List[str] = [
        "--skip_leading_rows=1",
        "--field_delimiter='\t'",
        "--source_format=CSV",
    ]
    flags: str = ' '.join(flags)
    command: str = f"bq load {flags} {table} {csv_path} {schema}"
    return command
