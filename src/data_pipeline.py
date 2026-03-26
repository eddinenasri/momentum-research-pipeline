"""
data_pipeline.py
================
Download, clean, and compute returns for the S&P 500 universe.

Pipeline Overview
-----------------
1. Download  : fetch daily adjusted close prices from yfinance for all
               tickers that ever appeared in our universe (2010–2024).
               Downloads in batches to respect yfinance rate limits.

2. Clean     : handle missing values, remove stocks with insufficient
               history, winsorize extreme returns to cap data errors,
               and align all stocks to a common trading calendar.

3. Returns   : compute daily simple returns from clean prices.
               Save both prices and returns as Parquet files.

Why Parquet
-----------
Parquet is a binary columnar format. Compared to CSV it is ~10x faster
to read and ~5x smaller on disk for numerical matrices. It is the
standard format in production data pipelines.

On Delisted Tickers
-------------------
Many of our 794 tickers are delisted (acquired, bankrupt, removed).
yfinance returns their full history up to the delisting date and then
stops no errors, just a shorter series. We include them because their
historical returns are needed to compute signals correctly.

Author: Eddine NASRI
"""

import os
import time
import logging

import numpy  as np
import pandas as pd
import yfinance as yf

from universe import get_all_tickers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Backtest window must match universe.py settings
START_DATE = "2010-01-01"
END_DATE   = "2024-01-01"

# Download settings
BATCH_SIZE  = 100    # tickers per yfinance request
BATCH_PAUSE = 1.0    # seconds between batches (respect rate limits)

# Cleaning settings
MIN_HISTORY_DAYS    = 252        # ~1 trading year minimum to compute signal
MAX_FORWARD_FILL    = 3          # max days to forward-fill missing prices
WINSOR_THRESHOLD    = 0.50       # cap daily returns at ±50%
MIN_DATA_COVERAGE   = 0.80       # drop stock if >20% of days are missing

# Output paths
RAW_PRICES_PATH  = "data/raw/prices_raw.parquet"
PRICES_PATH      = "data/processed/prices.parquet"
RETURNS_PATH     = "data/processed/returns.parquet"


# ---------------------------------------------------------------------------
# Step 1 - Download
# ---------------------------------------------------------------------------

def download_prices(
    tickers:    list[str],
    start_date: str = START_DATE,
    end_date:   str = END_DATE,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for all tickers.

    Downloads in batches of BATCH_SIZE to avoid yfinance rate limits.
    Tickers that return no data (delisted with no history in range) are
    silently skipped this is expected for some older tickers.

    We use auto_adjust=True which means yfinance returns prices already
    corrected for splits and dividends. We never touch raw unadjusted prices.

    Parameters
    ----------
    tickers : list[str]
        All tickers to download. Typically from get_all_tickers().
    start_date : str
        Start of the price history to download.
    end_date : str
        End of the price history to download.

    Returns
    -------
    pd.DataFrame
        Rows : trading days
        Columns : ticker symbols
        Values : adjusted closing prices (NaN where no data)
    """
    tickers = sorted(set(tickers))   # deduplicate and sort for reproducibility
    batches = [
        tickers[i : i + BATCH_SIZE]
        for i in range(0, len(tickers), BATCH_SIZE)
    ]

    logger.info(
        f"Downloading {len(tickers)} tickers in {len(batches)} batches "
        f"({start_date} → {end_date})"
    )

    all_prices: list[pd.DataFrame] = []

    for i, batch in enumerate(batches):
        try:
            raw = yf.download(
                batch,
                start      = start_date,
                end        = end_date,
                auto_adjust= True,     # adjusted prices: splits + dividends
                progress   = False,
                threads    = True,
            )

            # yfinance returns a MultiIndex when downloading multiple tickers.
            # We only want the "Close" level.
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                # Single ticker case reshape to DataFrame with ticker as column
                close = raw[["Close"]]
                close.columns = batch

            all_prices.append(close)
            logger.info(f"  Batch {i+1}/{len(batches)} sucess ({len(batch)} tickers)")

        except Exception as e:
            logger.warning(f"  Batch {i+1}/{len(batches)} failed - {e}")

        # Pause between batches to respect yfinance rate limits
        if i < len(batches) - 1:
            time.sleep(BATCH_PAUSE)

    if not all_prices:
        raise RuntimeError("No price data downloaded. Check your internet connection.")

    # Concatenate all batches along columns
    prices = pd.concat(all_prices, axis=1)

    # Remove duplicate columns (can occur if a ticker appears in two batches)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    logger.info(
        f"Download complete: {prices.shape[1]} tickers, "
        f"{prices.shape[0]} trading days"
    )
    return prices


# ---------------------------------------------------------------------------
# Step 2 - Clean
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw price matrix.

    Cleaning steps in order:

    1. Drop tickers with insufficient data coverage
       If more than 20% of a ticker's prices are missing, it is too
       unreliable to include. We drop the entire column.

    2. Align to common trading calendar
       Reindex all tickers to the union of all trading days, then
       forward-fill gaps up to MAX_FORWARD_FILL days. This ensures
       every stock has an entry for every trading day.

    3. Drop tickers with insufficient history
       After alignment, drop any ticker with fewer than MIN_HISTORY_DAYS
       non-null prices. We cannot compute a 12-month momentum signal
       on a stock with less than 1 year of history.

    4. Final forward-fill and drop remaining NaN
       A last pass to handle any edge cases, then drop columns still
       having NaN at the start (before the stock's IPO date).

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices from download_prices().

    Returns
    -------
    pd.DataFrame
        Clean price matrix ready for return computation.
    """
    logger.info(f"Cleaning prices: starting shape {prices.shape}")

    # --- Step 1 : Drop tickers with too many missing values ----------------
    # A ticker missing >20% of days is too unreliable to use.
    # threshold = minimum number of non-NaN values required to keep the column
    min_observations = int(len(prices) * MIN_DATA_COVERAGE)
    prices = prices.dropna(thresh=min_observations, axis=1)
    logger.info(f"  After coverage filter: {prices.shape[1]} tickers")

    # --- Step 2 : Align to common trading calendar -------------------------
    # The union of all dates across all tickers gives us the full calendar.
    # We forward-fill gaps (holidays, no-trade days) up to MAX_FORWARD_FILL.
    # This ensures all tickers share exactly the same set of dates.
    prices = prices.sort_index()
    prices = prices.ffill(limit=MAX_FORWARD_FILL)
    logger.info(f"  Forward-fill applied (max {MAX_FORWARD_FILL} days)")

    # --- Step 3 : Drop tickers with insufficient history -------------------
    # We need at least 1 trading year to compute a 12-month momentum signal.
    # Count non-NaN prices per ticker and filter.
    valid_history = prices.count() >= MIN_HISTORY_DAYS
    prices = prices.loc[:, valid_history]
    logger.info(f"  After history filter: {prices.shape[1]} tickers")

    # --- Step 4 : Final cleanup --------------------------------------------
    # One more forward-fill pass for any remaining gaps, then drop columns
    # that still have NaN at the very start (before IPO these are correct
    # NaN values, not data errors, so we keep them as NaN not filled).
    prices = prices.ffill(limit=MAX_FORWARD_FILL)

    logger.info(
        f"Cleaning complete: {prices.shape[1]} tickers, "
        f"{prices.shape[0]} trading days | "
        f"Remaining NaN: {prices.isna().sum().sum()}"
    )
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from clean prices.

    Simple return formula:
        r_t = (P_t - P_{t-1}) / P_{t-1}

    We use simple returns (not log returns) because:
    - Cross-sectional momentum uses multi-period returns computed
      by compounding: (1+r1)(1+r2)...(1+rN) - 1
    - Simple returns compound correctly across periods
    - Log returns are additive across time but not across assets

    After computing returns we winsorize to cap extreme values.
    A single-day return beyond ±WINSOR_THRESHOLD (50%) is almost
    certainly a data error for an S&P 500 stock. We cap it rather
    than removing it to preserve the direction of the move.

    Parameters
    ----------
    prices : pd.DataFrame
        Clean price matrix from clean_prices().

    Returns
    -------
    pd.DataFrame
        Daily simple returns. First row is NaN (no previous price).
        Shape: (trading_days - 1) × tickers
    """
    returns = prices.pct_change()

    # Drop the first row it is always NaN (no previous price to compare)
    returns = returns.iloc[1:]

    # Winsorize: cap returns at ±WINSOR_THRESHOLD
    # clip() replaces values below lower with lower, above upper with upper
    # This limits damage from data errors without removing observations
    returns_before = (returns.abs() > WINSOR_THRESHOLD).sum().sum()
    returns = returns.clip(lower=-WINSOR_THRESHOLD, upper=WINSOR_THRESHOLD)

    if returns_before > 0:
        logger.warning(
            f"Winsorized {returns_before} extreme return observations "
            f"(threshold: ±{WINSOR_THRESHOLD:.0%})"
        )

    logger.info(
        f"Returns computed: {returns.shape[0]} days × {returns.shape[1]} tickers"
    )
    return returns


# ---------------------------------------------------------------------------
# Step 3 - Save and Load
# ---------------------------------------------------------------------------

def save_data(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """
    Save clean prices and returns to Parquet files.

    Parquet is a binary columnar format ~10x faster to read and
    ~5x smaller on disk compared to CSV for numerical matrices.
    It preserves dtypes (float32, datetime index) without conversion.

    We save both prices and returns because:
    - Returns are used by signal.py for momentum computation
    - Prices are kept for reference and debugging
    """
    os.makedirs(os.path.dirname(PRICES_PATH),  exist_ok=True)
    os.makedirs(os.path.dirname(RETURNS_PATH), exist_ok=True)

    prices.to_parquet(PRICES_PATH)
    returns.to_parquet(RETURNS_PATH)

    prices_mb  = os.path.getsize(PRICES_PATH)  / 1e6
    returns_mb = os.path.getsize(RETURNS_PATH) / 1e6

    logger.info(f"Saved prices  → {PRICES_PATH}  ({prices_mb:.1f} MB)")
    logger.info(f"Saved returns → {RETURNS_PATH} ({returns_mb:.1f} MB)")


def load_returns(path: str = RETURNS_PATH) -> pd.DataFrame:
    """
    Load the precomputed returns matrix from disk.

    This is what signal.py will call it never re-downloads or
    re-computes, it just loads the already-processed returns.

    Returns
    -------
    pd.DataFrame
        Daily returns matrix. Rows: dates. Columns: tickers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Returns file not found at '{path}'. "
            f"Run data_pipeline.py first to download and process the data."
        )
    returns = pd.read_parquet(path)
    logger.info(
        f"Loaded returns: {returns.shape[0]} days × {returns.shape[1]} tickers"
    )
    return returns


def load_market_returns(path: str = "data/raw/market_returns.csv") -> pd.Series:
    """
    Load the S&P 500 daily returns series for dynamic weighting.

    Used by backtest.py to compute the bear market indicator (I_B)
    and realized market variance in Daniel & Moskowitz (2016) Eq. 6.
    Downloaded by data_pipeline.py during the initial setup run.

    Returns
    -------
    pd.Series
        Daily S&P 500 returns indexed by date.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Market returns not found at '{path}'. "
            f"Run data_pipeline.py first."
        )
    series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    series.name = "market_return"
    logger.info(f"Loaded market returns: {len(series)} days")
    return series


def load_prices(path: str = PRICES_PATH) -> pd.DataFrame:
    """
    Load the precomputed clean prices matrix from disk.

    Returns
    -------
    pd.DataFrame
        Clean adjusted prices. Rows: dates. Columns: tickers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prices file not found at '{path}'. "
            f"Run data_pipeline.py first to download and process the data."
        )
    prices = pd.read_parquet(path)
    logger.info(
        f"Loaded prices: {prices.shape[0]} days × {prices.shape[1]} tickers"
    )
    return prices


# ---------------------------------------------------------------------------
# Entry point run this file to execute the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Get the full ticker universe -----------------------------------
    logger.info("=== Step 1/3 : Building ticker universe ===")
    tickers = get_all_tickers(START_DATE, END_DATE)
    logger.info(f"Universe: {len(tickers)} unique tickers")

    # --- 2. Download raw prices --------------------------------------------
    logger.info("=== Step 2/3 : Downloading prices ===")
    logger.info("This will take 5–15 minutes. Grab a coffee ☕")
    raw_prices = download_prices(tickers, START_DATE, END_DATE)

    # Save raw prices immediately so if cleaning crashes we don't re-download
    os.makedirs("data/raw", exist_ok=True)
    raw_prices.to_parquet(RAW_PRICES_PATH)
    logger.info(f"Raw prices saved to {RAW_PRICES_PATH}")

    # --- 3. Clean and compute returns --------------------------------------
    logger.info("=== Step 3/3 : Cleaning and computing returns ===")
    clean  = clean_prices(raw_prices)
    returns = compute_returns(clean)

    # --- 4. Save final outputs ---------------------------------------------
    save_data(clean, returns)

    # --- 5. Download market index for dynamic weighting -------------------
    # The S&P 500 index (^GSPC) is needed for the bear market indicator
    # and realized market variance in Daniel & Moskowitz (2016) Eq. 6.
    # We start from 2008 so the 24-month lookback is available from 2010.
    logger.info("=== Downloading market index (^GSPC) ===")
    try:
        mkt_raw     = yf.download("^GSPC", start="2008-01-01", end=END_DATE,
                                   auto_adjust=True, progress=False)
        mkt_returns = mkt_raw["Close"].pct_change().dropna()
        mkt_returns.name = "market_return"
        mkt_returns.to_csv("data/raw/market_returns.csv", header=True)
        logger.info(
            f"Market returns saved: {len(mkt_returns)} days "
            f"({mkt_returns.index[0].date()} -> {mkt_returns.index[-1].date()})"
        )
    except Exception as e:
        logger.warning(f"Failed to download market index: {e}")

    # --- 6. Sanity checks --------------------------------------------------
    print("\n========== Pipeline Summary ==========")
    print(f"Raw tickers downloaded : {raw_prices.shape[1]}")
    print(f"Clean tickers retained : {clean.shape[1]}")
    print(f"Trading days           : {returns.shape[0]}")
    print(f"Date range             : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Missing values         : {returns.isna().sum().sum()}")
    print(f"\nSample returns (last 3 days, first 5 tickers):")
    print(returns.iloc[-3:, :5].round(4))
    print("======================================")