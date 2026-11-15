"""DuckDB connection with the project's resource limits."""

import duckdb

from . import config


def connect() -> duckdb.DuckDBPyConnection:
    config.ensure_dirs()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute(f"SET temp_directory = '{config.DUCKDB_TEMP_DIR.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    con.execute("SET enable_progress_bar = false")
    return con


def sql_path(path) -> str:
    return str(path).replace("'", "''")


def copy_to_parquet(con, query: str, target) -> None:
    """Write a query result to parquet via a temporary file, so an interrupted run leaves no partial output."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    con.execute(f"COPY ({query}) TO '{sql_path(partial)}' (FORMAT parquet, COMPRESSION zstd)")
    partial.replace(target)
