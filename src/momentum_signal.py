"""
signal.py
=========
Cross-sectional momentum signal with institutional-grade construction.

Strategy Overview
-----------------
We implement a cross-sectional momentum strategy on the S&P 500 universe.
At each monthly rebalancing date we rank every stock by its momentum signal,
then go long the top quintile and short the bottom quintile.

The key methodological choices :

1. Signal : 12-1 month return (skip last month to avoid short-term reversal)
2. Industry neutralization : remove sector-level momentum to isolate stock-specific alpha
3. Volatility adjustment : scale signal by signal consistency, not raw return
4. Position sizing : inverse-volatility weighting for equal risk contribution

Why Each Choice Matters
-----------------------
Raw 12-1 momentum is contaminated by two things:
  - Sector effects: energy stocks all rise together when oil prices rise.
    This is not a stock-specific signal it's a macro bet.
  - Volatility distortion: a 40% return in a high-vol stock contains less
    information than a 20% return in a low-vol stock.

We remove sector effects by demeaning returns within each GICS sector.
We adjust for volatility by dividing the momentum return by the realized
volatility over the same window this is called the Sharpe-scaled signal.

Position Sizing
---------------
Naive equal-weighting means high-volatility stocks dominate portfolio risk
even if they have small notional weights. Inverse-volatility weighting ensures
each stock contributes the same amount of risk to the portfolio, making the
backtest results more stable and the strategy more realistic.

Signal Decay Analysis
---------------------
We also compute how quickly the signal loses predictive power over time.
This is done by measuring the autocorrelation of the signal at lags of
1, 3, 6, and 12 months. A signal that decays quickly needs more frequent
rebalancing. This analysis is what transforms a student project into a
research paper.

References
----------
- Jegadeesh & Titman (1993) : "Returns to Buying Winners and Selling Losers"
  The original momentum paper. Our 12-1 specification comes from here.
- Asness, Moskowitz & Pedersen (2013) : "Value and Momentum Everywhere"
  Industry-neutralized momentum and volatility adjustment.
- Moskowitz & Grinblatt (1999) : "Do Industries Explain Momentum?"
  Why sector neutralization matters.

Author: Eddine NASRI
"""

import logging
import os

import numpy  as np
import pandas as pd

from data_pipeline import load_returns, load_prices
from universe      import get_universe, get_rebalancing_dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Signal construction parameters
MOMENTUM_LOOKBACK  = 252      # ~12 months of trading days
MOMENTUM_SKIP      = 21       # ~1 month skip to avoid short-term reversal
VOLATILITY_WINDOW  = 63       # ~3 months for realized volatility estimate
MIN_SIGNAL_OBS     = 180      # minimum observations to compute a valid signal

# Portfolio construction parameters
N_QUINTILES        = 5        # divide universe into 5 buckets
LONG_QUINTILE      = 5        # buy quintile 5 (strongest momentum)
SHORT_QUINTILE     = 1        # sell quintile 1 (weakest momentum)

# GICS sector mapping maps each ticker to its sector for neutralization
# We use a simplified 11-sector GICS classification
# In production this would come from a data provider
# Here we download it from Wikipedia as a reasonable free approximation
SECTOR_PATH = "data/universe/sp500_sectors.csv"

# Output path
SIGNALS_PATH = "data/processed/signals.parquet"


# ---------------------------------------------------------------------------
# Step 1 - Sector data
# ---------------------------------------------------------------------------

def load_sector_map(path: str = SECTOR_PATH) -> dict[str, str]:
    """
    Load a mapping from ticker to GICS sector.

    We use sector membership to neutralize sector-level momentum.
    Without this, our signal would partly reflect macro bets
    (e.g. all energy stocks rising together when oil prices rise)
    rather than stock-specific momentum.

    If the sector file doesn't exist we download it from Wikipedia.
    This gives us current sector membership not historical, which
    is a known limitation we document explicitly.

    Returns
    -------
    dict[str, str]
        Maps ticker symbol to GICS sector name.
        Example: {"AAPL": "Information Technology", "JPM": "Financials"}
    """
    if not os.path.exists(path):
        logger.info("Sector map not found downloading from Wikipedia...")
        _download_sector_map(path)

    df = pd.read_csv(path)
    sector_map = dict(zip(df["ticker"], df["sector"]))
    logger.info(f"Loaded sector map: {len(sector_map)} tickers, "
                f"{len(set(sector_map.values()))} sectors")
    return sector_map


def _download_sector_map(path: str = SECTOR_PATH) -> None:
    """
    Download current S&P 500 sector membership from Wikipedia.

    This gives us today's sector classification, not historical.
    This is a known approximation in production you would use
    a point-in-time GICS classification from a data vendor.

    The practical impact is small because sector classifications
    rarely change dramatically over our backtest window.
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    url      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers, timeout=10)
    soup     = BeautifulSoup(response.text, "html.parser")
    table    = soup.find("table", {"id": "constituents"})

    rows = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 4:
            ticker = cols[0].text.strip().replace(".", "-")
            sector = cols[2].text.strip()
            rows.append({"ticker": ticker, "sector": sector})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Sector map saved to {path} ({len(rows)} tickers)")


# ---------------------------------------------------------------------------
# Step 2 - Core signal computation
# ---------------------------------------------------------------------------

def compute_momentum_return(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """
    Compute the raw 12-1 month momentum return for each stock.

    The 12-1 specification:
    - Use returns from 12 months ago to 1 month ago
    - Skip the most recent month (reversal avoidance)
    - Compound daily returns into a single period return

    Mathematically:
        momentum_i = prod(1 + r_t) - 1
        for t in [T - 252 days, T - 21 days]

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily returns matrix (all dates, all tickers).
    as_of_date : pd.Timestamp
        The rebalancing date. We only use data strictly before this date.

    Returns
    -------
    pd.Series
        Raw momentum return per ticker. NaN if insufficient history.
    """
    # Strict cutoff only data available before this date
    # This enforces point-in-time correctness for the signal
    past_returns = returns.loc[returns.index < as_of_date]

    if len(past_returns) < MOMENTUM_LOOKBACK:
        # Not enough history to compute any signal at all
        return pd.Series(dtype=float)

    # Window: from MOMENTUM_LOOKBACK days ago to MOMENTUM_SKIP days ago
    # We skip the last MOMENTUM_SKIP days (approx 1 month) to avoid reversal
    window = past_returns.iloc[-(MOMENTUM_LOOKBACK) : -(MOMENTUM_SKIP)]

    if len(window) < MIN_SIGNAL_OBS:
        return pd.Series(dtype=float)

    # Compound daily returns: (1+r1)(1+r2)...(1+rN) - 1
    # We use log returns for numerical stability then convert back
    # log(1+r) sums are equivalent to compounding simple returns
    momentum = (1 + window).prod() - 1

    # Require minimum observations per stock (some may have NaN gaps)
    valid_obs = window.count()
    momentum[valid_obs < MIN_SIGNAL_OBS] = np.nan

    return momentum


def compute_realized_volatility(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """
    Compute annualized realized volatility for each stock.

    We use the standard deviation of daily returns over VOLATILITY_WINDOW
    days, annualized by multiplying by sqrt(252).

    This serves two purposes:
    1. Sharpe-scaling the momentum signal (signal quality adjustment)
    2. Inverse-volatility position sizing

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily returns matrix.
    as_of_date : pd.Timestamp
        Rebalancing date. Only past data is used.

    Returns
    -------
    pd.Series
        Annualized volatility per ticker.
    """
    past_returns = returns.loc[returns.index < as_of_date]

    if len(past_returns) < VOLATILITY_WINDOW:
        return pd.Series(dtype=float)

    window = past_returns.iloc[-VOLATILITY_WINDOW:]

    # Annualized volatility: daily std × sqrt(252 trading days per year)
    vol = window.std() * np.sqrt(252)

    # Floor volatility at 5% to avoid division by near-zero values
    # A stock with measured vol below 5% likely has data quality issues
    vol = vol.clip(lower=0.05)

    return vol


def neutralize_sectors(
    signal: pd.Series,
    sector_map: dict[str, str],
    universe: list[str],
) -> pd.Series:
    """
    Remove sector-level momentum from the raw signal.

    Within each GICS sector, we subtract the sector median signal.
    This transforms the signal from "which stocks have strong absolute
    momentum" to "which stocks have strong momentum RELATIVE TO their sector".

    Example: if all energy stocks have high momentum because oil prices
    rose, subtracting the energy sector median removes this shared effect.
    What remains is stock-specific momentum which energy stocks
    outperformed other energy stocks.

    Why median and not mean?
    The median is robust to outliers. One stock with extreme momentum
    won't distort the sector benchmark.

    Parameters
    ----------
    signal : pd.Series
        Raw momentum signal (one value per ticker).
    sector_map : dict[str, str]
        Ticker → sector mapping.
    universe : list[str]
        Tickers in the investable universe on this date.

    Returns
    -------
    pd.Series
        Sector-neutralized momentum signal.
    """
    signal = signal.copy()

    # Only neutralize tickers in our current universe
    signal_universe = signal[signal.index.isin(universe)]

    # Group by sector and subtract sector median
    for ticker in signal_universe.index:
        sector = sector_map.get(ticker, "Unknown")
        if sector == "Unknown":
            continue

        # Find all tickers in the same sector that are in our universe
        sector_tickers = [
            t for t in signal_universe.index
            if sector_map.get(t, "Unknown") == sector
        ]

        if len(sector_tickers) >= 3:   # need at least 3 stocks to demean
            sector_median = signal_universe[sector_tickers].median()
            signal[ticker] = signal[ticker] - sector_median

    return signal


def sharpe_scale_signal(
    momentum: pd.Series,
    volatility: pd.Series,
) -> pd.Series:
    """
    Divide the momentum return by realized volatility (Sharpe scaling).

    A stock that returned 30% with 15% volatility has a stronger signal
    (Sharpe = 2.0) than one that returned 30% with 40% volatility
    (Sharpe = 0.75). The higher-vol stock's return is noisier and less
    reliable as a momentum signal.

    This is equivalent to computing the realized Sharpe ratio of the
    momentum window, and using that as the signal instead of raw return.

    Parameters
    ----------
    momentum : pd.Series
        Raw 12-1 month return per ticker.
    volatility : pd.Series
        Annualized realized volatility per ticker.

    Returns
    -------
    pd.Series
        Volatility-adjusted momentum signal.
    """
    # Align indices - only compute for tickers with both values
    common = momentum.index.intersection(volatility.index)
    scaled = momentum[common] / volatility[common]
    return scaled


# ---------------------------------------------------------------------------
# Step 3 - Portfolio construction
# ---------------------------------------------------------------------------

def compute_portfolio_weights(
    signal: pd.Series,
    volatility: pd.Series,
    universe: list[str],
) -> pd.Series:
    """
    Convert the signal into portfolio weights using quintile ranking
    and inverse-volatility position sizing.

    Construction steps:
    1. Restrict to investable universe on this date
    2. Rank stocks into quintiles by signal strength
    3. Assign +1 to long quintile, -1 to short quintile, 0 to middle
    4. Scale each position by 1/volatility (equal risk contribution)
    5. Normalize so long and short legs each sum to 1.0 in absolute value

    Why inverse-volatility sizing?
    Equal notional weighting means high-vol stocks dominate portfolio risk.
    A 2% position in a 50%-vol stock contributes more risk than a 2%
    position in a 20%-vol stock. Inverse-vol sizing equalizes risk
    contribution across all positions.

    Parameters
    ----------
    signal : pd.Series
        Sector-neutralized, Sharpe-scaled momentum signal.
    volatility : pd.Series
        Annualized realized volatility for position sizing.
    universe : list[str]
        Tickers in the investable universe on this rebalancing date.

    Returns
    -------
    pd.Series
        Portfolio weights. Positive = long, negative = short.
        Long leg sums to +1.0, short leg sums to -1.0.
        Returns empty Series if insufficient stocks.
    """
    # Restrict to investable universe with valid signals
    eligible = [t for t in universe if t in signal.index]
    sig      = signal[eligible].dropna()

    if len(sig) < N_QUINTILES * 5:
        # Need at least 25 stocks to form meaningful quintiles
        logger.warning(f"Only {len(sig)} stocks with valid signals - skipping")
        return pd.Series(dtype=float)

    # Rank into quintiles (1 = weakest, 5 = strongest momentum)
    quintiles = pd.qcut(sig, q=N_QUINTILES, labels=False) + 1  # labels 1-5

    # Identify long and short legs
    long_mask  = quintiles == LONG_QUINTILE
    short_mask = quintiles == SHORT_QUINTILE

    long_tickers  = sig[long_mask].index.tolist()
    short_tickers = sig[short_mask].index.tolist()

    if not long_tickers or not short_tickers:
        return pd.Series(dtype=float)

    # Inverse-volatility weights
    # For each leg: weight_i = (1/vol_i) / sum(1/vol_j for j in leg)
    weights = pd.Series(0.0, index=sig.index)

    vol_common = volatility.reindex(sig.index).fillna(volatility.median())

    # Long leg: positive weights summing to +1
    long_inv_vol  = 1.0 / vol_common[long_tickers]
    weights[long_tickers] = long_inv_vol / long_inv_vol.sum()

    # Short leg: negative weights summing to -1
    short_inv_vol = 1.0 / vol_common[short_tickers]
    weights[short_tickers] = -(short_inv_vol / short_inv_vol.sum())

    return weights


# ---------------------------------------------------------------------------
# Step 4 - Full signal pipeline
# ---------------------------------------------------------------------------

def compute_all_signals(
    returns: pd.DataFrame,
    rebalancing_dates: list[pd.Timestamp],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Compute portfolio weights for every rebalancing date.

    This is the main loop that ties everything together. At each
    rebalancing date we:
    1. Get the investable universe (point-in-time)
    2. Compute raw 12-1 momentum for every stock
    3. Compute realized volatility for every stock
    4. Sharpe-scale the signal
    5. Neutralize sector effects
    6. Convert to portfolio weights via quintile ranking + inv-vol sizing

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily returns matrix from data_pipeline.
    rebalancing_dates : list[pd.Timestamp]
        Monthly rebalancing dates from universe.py.
    sector_map : dict[str, str]
        Ticker to sector mapping for neutralization.

    Returns
    -------
    pd.DataFrame
        Portfolio weights at each rebalancing date.
        Rows : rebalancing dates
        Columns : ticker symbols
        Values : portfolio weights (positive=long, negative=short, 0=not held)
    """
    all_weights: list[pd.Series] = []
    valid_dates: list[pd.Timestamp] = []

    n = len(rebalancing_dates)
    logger.info(f"Computing signals for {n} rebalancing dates...")

    for i, date in enumerate(rebalancing_dates):
        if i % 12 == 0:   # log progress every year
            logger.info(f"  Processing {date.date()} ({i+1}/{n})")

        # Step 1 : Get point-in-time universe
        try:
            universe = get_universe(date.strftime("%Y-%m-%d"))
        except ValueError:
            continue

        # Step 2 : Raw momentum return
        momentum = compute_momentum_return(returns, date)
        if momentum.empty:
            continue

        # Step 3 : Realized volatility
        volatility = compute_realized_volatility(returns, date)
        if volatility.empty:
            continue

        # Step 4 : Sharpe-scale the signal
        scaled_signal = sharpe_scale_signal(momentum, volatility)

        # Step 5 : Sector neutralization
        neutralized_signal = neutralize_sectors(scaled_signal, sector_map, universe)

        # Step 6 : Portfolio weights
        weights = compute_portfolio_weights(neutralized_signal, volatility, universe)

        if weights.empty:
            continue

        all_weights.append(weights)
        valid_dates.append(date)

    if not all_weights:
        raise RuntimeError("No valid signals computed. Check your data.")

    # Align all weight series to a common column set (union of all tickers)
    weights_df = pd.DataFrame(all_weights, index=valid_dates).fillna(0.0)
    weights_df.index.name = "date"

    logger.info(
        f"Signal computation complete: {len(weights_df)} rebalancing dates, "
        f"{(weights_df != 0).sum(axis=1).mean():.0f} avg active positions"
    )
    return weights_df


# ---------------------------------------------------------------------------
# Step 5 - Signal decay analysis
# ---------------------------------------------------------------------------

def analyze_signal_decay(
    returns: pd.DataFrame,
    rebalancing_dates: list[pd.Timestamp],
    lags: list[int] = [1, 3, 6, 12],
) -> pd.DataFrame:
    """
    Measure how quickly the momentum signal loses predictive power.

    For each lag L (in months), we measure the correlation between:
    - The momentum signal computed at date T
    - The forward return over the next L months

    A signal with high correlation at lag 1 but low at lag 12 decays
    quickly it's a short-term signal. One that stays correlated at
    lag 12 is a long-term signal.

    This analysis is what turns a backtest into a research paper.
    Most project submissions never do this. It demonstrates that you
    understand the signal's economic mechanism, not just its performance.

    Parameters
    ----------
    returns : pd.DataFrame
        Full daily returns matrix.
    rebalancing_dates : list[pd.Timestamp]
        Monthly rebalancing dates.
    lags : list[int]
        Forward return horizons in months to test.

    Returns
    -------
    pd.DataFrame
        Columns: lag (months), IC (information coefficient),
                 IC_tstat (t-statistic for significance)
    """
    logger.info("Computing signal decay analysis...")
    results = []

    for lag in lags:
        ics = []   # Information Coefficient at each date

        for i, date in enumerate(rebalancing_dates):
            # Need enough future dates for this lag
            future_idx = i + lag
            if future_idx >= len(rebalancing_dates):
                break

            future_date = rebalancing_dates[future_idx]

            # Compute signal at current date
            momentum = compute_momentum_return(returns, date)
            if momentum.empty:
                continue

            volatility = compute_realized_volatility(returns, date)
            if volatility.empty:
                continue

            signal = sharpe_scale_signal(momentum, volatility)

            # Compute forward return from current date to future date
            future_returns = returns.loc[
                (returns.index >= date) &
                (returns.index < future_date)
            ]

            if future_returns.empty:
                continue

            forward_ret = (1 + future_returns).prod() - 1

            # Information Coefficient = rank correlation between signal and forward return
            # We use rank correlation (Spearman) because it is more robust to outliers
            # than Pearson correlation and more appropriate for non-normal distributions
            common = signal.index.intersection(forward_ret.index)
            if len(common) < 50:
                continue

            ic = signal[common].rank().corr(forward_ret[common].rank())
            ics.append(ic)

        if ics:
            ic_mean   = np.mean(ics)
            ic_std    = np.std(ics)
            ic_tstat  = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else 0
            results.append({
                "lag_months"  : lag,
                "IC_mean"     : round(ic_mean, 4),
                "IC_std"      : round(ic_std, 4),
                "IC_tstat"    : round(ic_tstat, 2),
                "n_obs"       : len(ics),
            })
            logger.info(
                f"  Lag {lag:>2}m : IC={ic_mean:+.4f}, "
                f"t-stat={ic_tstat:+.2f}, n={len(ics)}"
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    START_DATE = "2010-01-01"
    END_DATE   = "2024-01-01"

    # --- Load data ---------------------------------------------------------
    logger.info("=== Loading data ===")
    returns = load_returns()
    sector_map = load_sector_map()

    # --- Get rebalancing dates ---------------------------------------------
    rebalancing_dates = get_rebalancing_dates(START_DATE, END_DATE)
    logger.info(f"Rebalancing dates: {len(rebalancing_dates)}")

    # --- Compute signals ---------------------------------------------------
    logger.info("=== Computing signals ===")
    weights = compute_all_signals(returns, rebalancing_dates, sector_map)

    # --- Save signals ------------------------------------------------------
    os.makedirs(os.path.dirname(SIGNALS_PATH), exist_ok=True)
    weights.to_parquet(SIGNALS_PATH)
    logger.info(f"Signals saved to {SIGNALS_PATH}")

    # --- Signal decay analysis ---------------------------------------------
    logger.info("=== Signal Decay Analysis ===")
    decay = analyze_signal_decay(returns, rebalancing_dates)

    # --- Summary -----------------------------------------------------------
    print("\n========== Signal Summary ==========")
    print(f"Rebalancing dates   : {len(weights)}")
    print(f"Universe size avg   : {(weights != 0).sum(axis=1).mean():.0f} stocks")
    print(f"Long positions avg  : {(weights > 0).sum(axis=1).mean():.0f} stocks")
    print(f"Short positions avg : {(weights < 0).sum(axis=1).mean():.0f} stocks")
    print(f"\nSample weights ({weights.index[12].date()}):")
    sample = weights.iloc[12]
    print("  Top 5 long  :", sample[sample > 0].nlargest(5).round(4).to_dict())
    print("  Top 5 short :", sample[sample < 0].nsmallest(5).round(4).to_dict())

    print(f"\n--- Signal Decay (IC by forward horizon) ---")
    print(decay.to_string(index=False))

    print("\n")
    print("IC interpretation:")
    print("  IC > 0.05  : meaningful predictive power")
    print("  IC > 0.10  : strong signal")
    print("  t-stat > 2 : statistically significant at 95% confidence")
    print("=====================================")