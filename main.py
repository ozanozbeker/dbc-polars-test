"""Try the adbc driver with various methods, timed repeatedly."""

import time
from collections.abc import Callable

import polars as pl
from adbc_driver_manager import dbapi
from rich import print

QUERY = """
    SELECT
        spc_latin,
        spc_common,
        COUNT(*) AS count,
        ROUND(COUNTIF(health="Good")/COUNT(*)*100) AS healthy_pct
    FROM
        `bigquery-public-data.new_york.tree_census_2015`
    WHERE
        status="Alive"
    GROUP BY
        spc_latin,
        spc_common
    ORDER BY
        count DESC
"""

N_RUNS = 30


def dbc_docs() -> None:
    """Try polars with docs method."""
    with (
        dbapi.connect(
            driver="bigquery",
            db_kwargs={
                "adbc.bigquery.sql.project_id": "doc-polars-test",
                "adbc.bigquery.sql.dataset_id": "bigquery-public-data",
            },
        ) as con,
        con.cursor() as cursor,
    ):
        cursor.execute(QUERY)
        tbl = cursor.fetch_arrow_table()
        _ = pl.DataFrame(tbl)


def pl_direct_con(*, close_con: bool = True) -> None:
    """Try polars with `con`."""
    con = dbapi.connect(
        driver="bigquery",
        db_kwargs={
            "adbc.bigquery.sql.project_id": "doc-polars-test",
            "adbc.bigquery.sql.dataset_id": "bigquery-public-data",
        },
    )
    _ = pl.read_database(query=QUERY, connection=con)
    if close_con:
        con.close()


def pl_direct_cursor(*, close_con: bool = True) -> None:
    """Try polars with `con.cursor()`."""
    con = dbapi.connect(
        driver="bigquery",
        db_kwargs={
            "adbc.bigquery.sql.project_id": "doc-polars-test",
            "adbc.bigquery.sql.dataset_id": "bigquery-public-data",
        },
    )
    _ = pl.read_database(query=QUERY, connection=con.cursor())
    if close_con:
        con.close()


def measure(fn: Callable) -> float:
    """Return elapsed time in seconds."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main():
    "Run tests."
    records_close = []
    records_leak = []

    # Phase 1 - Properly closed connections
    for f in (dbc_docs, pl_direct_con, pl_direct_cursor):
        f()  # warm-up for auth
        for i in range(N_RUNS):
            payload = {
                "method": f.__name__,
                "phase": "closed",
                "run": i + 1,
                "time_sec": measure(f),
            }
            print(
                f"[bold cyan]{payload['method']}[/bold cyan] "
                f"[green]{payload['phase']}[/green] "
                f"run {i + 1}/{N_RUNS}: {payload['time_sec']:.3f} sec"
            )

            records_close.append(payload)

    # Phase 2 - No-close variants (simulate doc-style leakage)
    for f in (pl_direct_con, pl_direct_cursor):
        f()  # warm-up for auth
        for i in range(N_RUNS):
            payload = {
                "method": f.__name__,
                "phase": "not_closed",
                "run": i + 1,
                "time_sec": measure(lambda f=f: f(close_con=False)),
            }
            print(
                f"[bold cyan]{payload['method']}[/bold cyan] "
                f"[green]{payload['phase']}[/green] "
                f"run {i + 1}/{N_RUNS}: {payload['time_sec']:.3f} sec"
            )
            records_leak.append(payload)

    df = pl.concat([pl.DataFrame(records_close), pl.DataFrame(records_leak)])
    summary = (
        df.group_by(["method", "phase"])
        .agg(
            pl.col("time_sec").mean().alias("mean"),
            pl.col("time_sec").std().alias("std"),
            pl.col("time_sec").min().alias("min"),
            pl.col("time_sec").max().alias("max"),
        )
        .sort("mean")
    )

    df.write_csv("runs.csv")
    summary.write_csv("summary.csv")

    print("\nSummary (sec):")
    print(summary)


if __name__ == "__main__":
    main()
