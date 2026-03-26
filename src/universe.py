"""
universe.py
===========
Point-in-time S&P 500 universe lookup.

Responsibility
--------------
This module answers one question for the rest of the codebase:
"Which stocks were in the S&P 500 on a given date?"

It is a clean interface over the historical constituents CSV.
Every other module (data_pipeline, signal, backtest) calls
get_universe(date) and never touches the CSV directly.
This means if the data source ever changes, only this file
needs to be updated encapsulation in practice.

Data Source
-----------
S&P 500 Historical Components & Changes
https://github.com/fja05680/sp500

Originally sourced from Andreas Clenow's "Trading Evolved",
extended and maintained by the open-source community.
More complete than Wikipedia's change log, especially pre-2015.

CSV Format
----------
The file has two columns:
    date    : snapshot date (one row per trading day)
    tickers : all tickers as one comma-separated string

Tickers with date suffixes (e.g. AGN-201503) encode the month
the stock was removed from the index. We strip these suffixes
to recover clean tradeable symbols (AGN-201503 → AGN).

Survivorship Bias
-----------------
The CSV already handles survivorship bias each row contains
only the stocks alive in the index on that specific date.
This module enforces point-in-time correctness by always
returning the composition as of the requested date, never future.

Author: Eddine NASRI
"""

import re
import os
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/universe/sp500_historical_components.csv"

# Module-level cache the CSV is loaded once and reused for every subsequent call.
# None means "not loaded yet". This is the simplest form of memoization.
_CONSTITUENTS_CACHE: dict | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_removal_date(ticker: str) -> str:
    """
    Strip the removal date suffix from a ticker symbol.

    The CSV encodes removal dates directly in ticker names:
        'AAL-199702'  → 'AAL'   (removed February 1997)
        'AGN-201503'  → 'AGN'   (removed March 2015)
        'AAPL'        → 'AAPL'  (still in index, no suffix)

    We only strip 6-digit numeric suffixes (YYYYMM format).
    Hyphens that are part of the real ticker (e.g. BRK-B) are
    preserved because they are followed by a letter, not digits.

    Parameters
    ----------
    ticker : str
        Raw ticker string from the CSV.

    Returns
    -------
    str
        Clean ticker symbol without date suffix.
    """
    return re.sub(r"-\d{6}$", "", ticker.strip())


def _load_constituents(path: str = DATA_PATH) -> dict[str, list[str]]:
    """
    Load and parse the historical constituents CSV.

    The CSV has exactly two columns:
        date    : composition snapshot date
        tickers : all tickers as one comma-separated string

    Raw example row:
        1996-01-02,"AAPL,AGN-201503,TMC-200006,GE,..."

    We parse each tickers string, split on commas, strip removal
    date suffixes, and deduplicate to get clean ticker lists.

    Returns
    -------
    dict[str, list[str]]
        Maps each date string "YYYY-MM-DD" to a list of clean
        ticker symbols. Sorted by date ascending.
    """
    global _CONSTITUENTS_CACHE
    if _CONSTITUENTS_CACHE is not None:
        return _CONSTITUENTS_CACHE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Historical constituents file not found at '{path}'.\n"
            f"Download it with:\n"
            f"  curl -L \"https://raw.githubusercontent.com/fja05680/sp500"
            f"/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv\""
            f" -o \"{path}\""
        )

    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    constituents: dict[str, list[str]] = {}

    for _, row in raw.iterrows():
        date_str    = row["date"].strftime("%Y-%m-%d")
        raw_tickers = str(row["tickers"]).split(",")

        # Strip removal date suffixes from each ticker
        clean = [_strip_removal_date(t) for t in raw_tickers if t.strip()]

        # Deduplicate while preserving order
        # (stripping suffixes can create duplicates e.g. AGN and AGN-201503)
        seen:   set[str]  = set()
        deduped: list[str] = []
        for t in clean:
            if t and t not in seen:
                seen.add(t)
                deduped.append(t)

        constituents[date_str] = deduped

    dates = list(constituents.keys())
    logger.info(
        f"Loaded universe: {len(constituents)} dates "
        f"from {dates[0]} to {dates[-1]}"
    )
    _CONSTITUENTS_CACHE = constituents
    return constituents


# ---------------------------------------------------------------------------
# Public API - the only functions other modules should call
# ---------------------------------------------------------------------------

def get_universe(date: str | pd.Timestamp, path: str = DATA_PATH) -> list[str]:
    """
    Return the S&P 500 tickers as of a given date.

    Returns the composition from the most recent available date
    that is ≤ the requested date. This enforces point-in-time
    correctness - you only know the last published composition,
    never a future one.

    Parameters
    ----------
    date : str or pd.Timestamp
        The date for which you want the universe.
        Example: "2015-06-30"

    Returns
    -------
    list[str]
        Clean ticker symbols in the S&P 500 as of that date.

    Raises
    ------
    ValueError
        If the requested date is before the earliest available date.

    Example
    -------
    >>> tickers = get_universe("2012-01-31")
    >>> "AAPL" in tickers
    True
    """
    constituents = _load_constituents(path)
    date         = pd.Timestamp(date)
    all_dates    = sorted(constituents.keys())

    # Find the most recent available date that is <= requested date
    past_dates = [d for d in all_dates if pd.Timestamp(d) <= date]

    if not past_dates:
        raise ValueError(
            f"No universe data available on or before {date.date()}. "
            f"Earliest available date is {all_dates[0]}."
        )

    closest = past_dates[-1]
    tickers = constituents[closest]

    logger.debug(f"Universe on {closest}: {len(tickers)} stocks")
    return tickers


def get_all_tickers(
    start_date: str,
    end_date: str,
    path: str = DATA_PATH,
) -> list[str]:
    """
    Return every unique ticker that appeared in the S&P 500
    between start_date and end_date (inclusive).

    Used by data_pipeline.py to determine the full set of stocks
    it needs to download prices for every stock that ever existed
    in our universe, not just today's survivors.

    Parameters
    ----------
    start_date : str
        Start of the backtest period. Example: "2010-01-01"
    end_date : str
        End of the backtest period.   Example: "2024-01-01"

    Returns
    -------
    list[str]
        Deduplicated, sorted list of all tickers in the window.

    Example
    -------
    >>> tickers = get_all_tickers("2010-01-01", "2024-01-01")
    >>> len(tickers)  # Expect ~800
    """
    constituents = _load_constituents(path)
    start        = pd.Timestamp(start_date)
    end          = pd.Timestamp(end_date)

    all_tickers: set[str] = set()

    for date_str, tickers in constituents.items():
        d = pd.Timestamp(date_str)
        if start <= d <= end:
            all_tickers.update(tickers)

    result = sorted(all_tickers)
    logger.info(
        f"Total unique tickers ({start_date} → {end_date}): {len(result)}"
    )
    return result


def get_rebalancing_dates(
    start_date: str,
    end_date:   str,
    frequency:  str = "monthly",
    path:       str = DATA_PATH,
) -> list[pd.Timestamp]:
    """
    Return rebalancing dates between start and end.

    Parameters
    ----------
    start_date : str
    end_date : str
    frequency : str
        "monthly" (default), "quarterly", or "daily".

    Returns
    -------
    list[pd.Timestamp]
    """
    constituents = _load_constituents(path)
    start        = pd.Timestamp(start_date)
    end          = pd.Timestamp(end_date)

    all_dates = pd.DatetimeIndex(sorted([
        pd.Timestamp(d)
        for d in constituents.keys()
        if start <= pd.Timestamp(d) <= end
    ]))

    if frequency == "daily":
        return all_dates.tolist()

    alias = "ME" if frequency == "monthly" else "QE"
    date_series = pd.Series(all_dates, index=all_dates)
    resampled   = date_series.resample(alias).last().dropna()
    dates       = [pd.Timestamp(d) for d in resampled.values]

    logger.info(f"Rebalancing dates ({start_date} -> {end_date}, {frequency}): {len(dates)}")
    return dates

# ---------------------------------------------------------------------------
# Entry point run directly to sanity check the data
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    START_DATE = "2010-01-01"
    END_DATE   = "2024-01-01"

    print("\n========== Universe Sanity Check ==========")

    # 1. Spot check specific dates
    test_dates = ["2010-06-30", "2015-01-31", "2020-03-31", "2023-12-31"]
    for date in test_dates:
        tickers = get_universe(date)
        print(f"{date} : {len(tickers):>3} stocks | first 5: {tickers[:5]}")

    # 2. Full ticker universe across backtest window
    all_tickers = get_all_tickers(START_DATE, END_DATE)
    print(f"\nAll unique tickers ({START_DATE} → {END_DATE}): {len(all_tickers)}")

    # 3. Sanity check: AAPL should always be in the index
    for date in test_dates:
        tickers = get_universe(date)
        assert "AAPL" in tickers, f"AAPL missing on {date}  something is wrong"
    print("AAPL present on all test dates ✅")

    # 4. Rebalancing dates
    dates = get_rebalancing_dates(START_DATE, END_DATE)
    print(f"\nRebalancing dates : {len(dates)}")
    print(f"First : {dates[0].date()} | Last : {dates[-1].date()}")
    print("===========================================")
