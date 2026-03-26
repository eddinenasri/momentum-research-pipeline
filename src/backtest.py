"""
backtest.py
===========
Portfolio simulation engine for the cross-sectional momentum strategy.

Execution Model
---------------
Signal computed at close of date T, executed at open of date T+1.
We approximate T+1 open with T+1 close standard when intraday
data is unavailable.

Transaction Costs
-----------------
10 basis points one-way on actual turnover only. Unchanged positions
incur no cost.

Volatility Scaling
------------------
Implements constant volatility scaling from Daniel & Moskowitz (2016).
At each rebalancing date, the entire WML portfolio is scaled so that
its annualized volatility targets a fixed level (default 10%).

    scale = target_vol / realized_vol_past_126_days

When recent portfolio volatility is high which precedes momentum
crashes exposure is automatically reduced. When volatility is low,
exposure is increased up to a cap of 2x leverage.

This is position sizing discipline, not hedging. The signal (which
stocks to buy/sell) is unchanged. Only the total capital deployed
in the strategy varies over time.

Walk-Forward Validation
-----------------------
Sequential fold testing to assess consistency of performance across
different market regimes. A robust strategy should show positive
Sharpe ratios across most folds, not just concentrate returns in
one lucky period.

References
----------
- Daniel & Moskowitz (2016): Momentum Crashes, JFE 122(2), 221-247
- Lopez de Prado (2018): Advances in Financial Machine Learning
- Grinold & Kahn (1999): Active Portfolio Management

Author: Eddine NASRI
"""

import os
import logging

import numpy  as np
import pandas as pd

from data_pipeline   import load_returns, load_market_returns
from momentum_signal import compute_all_signals, load_sector_map
from universe        import get_rebalancing_dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

START_DATE = "2010-01-01"
END_DATE   = "2024-01-01"

# Transaction costs
COST_BPS  = 10
COST_RATE = COST_BPS / 10_000

# Borrowing costs
BASE_BORROW_COST = 0.02  # 200 bps

# Volatility scaling (Daniel & Moskowitz 2016)
VOL_TARGET = 0.10   # 10% annualized target volatility
VOL_WINDOW = 126    # 126 trading days ≈ 6 months (per the paper)
VOL_CAP    = 2.0    # maximum leverage never more than 2x

# Walk-forward validation
N_FOLDS = 5

# Output paths
PORTFOLIO_PATH = "data/processed/portfolio_returns.parquet"
METRICS_PATH   = "results/metrics.csv"


# ---------------------------------------------------------------------------
# Volatility scaling
# ---------------------------------------------------------------------------

def _compute_vol_scale(
    past_returns: list[float],
    target_vol:   float = VOL_TARGET,
    window:       int   = VOL_WINDOW,
    cap:          float = VOL_CAP,
) -> float:
    """
    Compute the constant volatility scaling factor for the WML portfolio.

    Uses only past returns never future so there is no look-ahead bias.
    Returns 1.0 (no scaling) when there is insufficient history.

    Parameters
    ----------
    past_returns : list[float]
        Daily WML gross returns up to but not including today.
    target_vol : float
        Annualized volatility target.
    window : int
        Lookback in trading days.
    cap : float
        Maximum scaling factor (caps leverage).

    Returns
    -------
    float
        Scaling factor in [0.0, cap].
    """
    if len(past_returns) < window:
        return 1.0

    recent       = np.array(past_returns[-window:])
    realized_vol = recent.std() * np.sqrt(252)

    if realized_vol < 1e-8:
        return 1.0

    return float(np.clip(target_vol / realized_vol, 0.0, cap))



# ---------------------------------------------------------------------------
# Dynamic weighting (Daniel & Moskowitz 2016, Section 4)
# ---------------------------------------------------------------------------

def _compute_bear_market_indicator(
    market_returns: list[float],
    window:         int = 504,   # 24 months ≈ 504 trading days
) -> int:
    """
    Bear market indicator I_B from Daniel & Moskowitz (2016).

    Returns 1 if the cumulative market return over the past 24 months
    is negative, 0 otherwise. This is the key conditioning variable
    that identifies panic states where momentum crashes occur.

    Parameters
    ----------
    market_returns : list[float]
        Daily market (benchmark) returns up to but not including today.
    window : int
        Lookback in trading days. 504 ≈ 24 months.

    Returns
    -------
    int
        1 if bear market, 0 otherwise.
    """
    if len(market_returns) < window:
        return 0
    recent      = np.array(market_returns[-window:])
    cumul_return = (1 + recent).prod() - 1
    return int(cumul_return < 0)


def _compute_dynamic_scale(
    past_wml_returns:    list[float],
    past_market_returns: list[float],
    target_vol:          float = VOL_TARGET,
    vol_window:          int   = VOL_WINDOW,
    bear_window:         int   = 504,
    cap:                 float = VOL_CAP,
    min_obs_regression:  int   = 60,
) -> float:
    """
    Dynamic WML portfolio scaling factor (Daniel & Moskowitz 2016, Eq. 6).

    Computes the optimal Markowitz weight by scaling position size
    proportionally to the ratio of forecasted expected return to
    forecasted variance:

        w* = (1/2λ) × μ_{t-1} / σ²_{t-1}

    The conditional expected return is forecasted using a rolling OLS
    regression of past WML returns on the interaction between the bear
    market indicator and realized market variance (Table 5, column 4):

        μ_{t-1} = γ₀ + γ_int × I_B × σ²_m

    All estimates use an expanding window strictly out-of-sample.
    This matches the "dyn, out-of-sample" implementation in Table 7.

    The result is then normalized so the unconditional volatility of
    the dynamic strategy matches VOL_TARGET, consistent with the paper.

    Parameters
    ----------
    past_wml_returns : list[float]
        Daily WML gross returns up to but not including today.
    past_market_returns : list[float]
        Daily benchmark returns up to but not including today.
    target_vol : float
        Annualized volatility target for normalization.
    vol_window : int
        Lookback for realized variance estimates (126 days = 6 months).
    bear_window : int
        Lookback for bear market indicator (504 days = 24 months).
    cap : float
        Maximum absolute scaling factor.
    min_obs_regression : int
        Minimum observations before fitting the regression.

    Returns
    -------
    float
        Dynamic scaling factor clipped to [-cap, cap].
        Negative values mean the strategy is reversed (short momentum).
    """
    n = len(past_wml_returns)

    # Need sufficient history for both the regression and variance estimates
    if n < max(bear_window, vol_window, min_obs_regression):
        # Fall back to constant vol scaling until we have enough data
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    wml    = np.array(past_wml_returns)
    market = np.array(past_market_returns[:n])

    # --- Build the regression dataset ---
    # For each past month t, compute:
    #   y_t     = WML return in month t (we use monthly aggregates)
    #   I_B     = bear market indicator as of month t-1
    #   σ²_m    = realized market variance over 126 days prior to month t

    # Approximate monthly by taking every 21-day window
    step     = 21
    indices  = list(range(vol_window, n - step, step))

    if len(indices) < min_obs_regression:
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    y_vals      = []
    interaction = []

    for idx in indices:
        # WML return over the next ~month
        y_month = (1 + wml[idx:idx+step]).prod() - 1

        # Bear market indicator at this point
        ib = int((1 + market[max(0, idx-bear_window):idx]).prod() - 1 < 0)

        # Realized market variance over prior 126 days
        mkt_window = market[max(0, idx-vol_window):idx]
        sigma2_m   = mkt_window.var() * 252   # annualized

        y_vals.append(y_month)
        interaction.append(ib * sigma2_m)

    y      = np.array(y_vals)
    X      = np.column_stack([np.ones(len(y)), interaction])

    # OLS: y = γ₀ + γ_int × (I_B × σ²_m)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        gamma_0   = coeffs[0]
        gamma_int = coeffs[1]
    except Exception:
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    # --- Forecast conditional expected return at current date ---
    ib_now      = _compute_bear_market_indicator(past_market_returns, bear_window)
    sigma2_m_now = np.array(past_market_returns[-vol_window:]).var() * 252
    mu_forecast  = gamma_0 + gamma_int * ib_now * sigma2_m_now

    # --- Forecast conditional variance of WML ---
    sigma2_wml = np.array(past_wml_returns[-vol_window:]).var() * 252
    if sigma2_wml < 1e-8:
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    # --- Optimal Markowitz weight: w* = (1/2λ) × μ_t / σ²_t  [D&M Eq. 6] ---
    #
    # We calibrate λ from the historical distribution of raw_scale = μ/σ²
    # so that the unconditional volatility of the strategy hits target_vol.
    # Closed-form solution: λ = sqrt(E[raw_scale² × σ²_wml]) / (2 × target_vol)
    # All estimates are expanding-window no look-ahead bias.

    raw_scales_hist = []
    sigma2_wml_hist = []

    for idx in indices:
        s2_wml = wml[max(0, idx - vol_window):idx].var() * 252
        if s2_wml < 1e-8:
            continue
        ib_hist   = int((1 + market[max(0, idx - bear_window):idx]).prod() - 1 < 0)
        s2_m_hist = market[max(0, idx - vol_window):idx].var() * 252
        mu_hist   = gamma_0 + gamma_int * ib_hist * s2_m_hist
        raw_scales_hist.append(mu_hist / s2_wml)
        sigma2_wml_hist.append(s2_wml)

    if len(raw_scales_hist) < 10:
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    rs  = np.array(raw_scales_hist)
    s2s = np.array(sigma2_wml_hist)
    lam = np.sqrt(np.mean(rs ** 2 * s2s)) / (2.0 * target_vol)

    if lam < 1e-8:
        return _compute_vol_scale(past_wml_returns, target_vol, vol_window, cap)

    scale = (mu_forecast / sigma2_wml) / (2.0 * lam)

    return float(np.clip(scale, -cap, cap))


# ---------------------------------------------------------------------------
# Compute Beta
# ---------------------------------------------------------------------------
def compute_rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int = 126):
    """
    Compute rolling beta of each stock vs market.
    """
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for t in range(window, len(returns)):
        window_returns = returns.iloc[t-window:t]
        window_market  = market.iloc[t-window:t]

        # demean
        r = window_returns - window_returns.mean()
        m = window_market - window_market.mean()

        # covariance: E[(r_i - mean)*(m - mean)]
        cov = (r.mul(m, axis=0)).mean()

        # variance of market
        var = (m ** 2).mean()

        if var > 1e-8:
            betas.iloc[t] = cov / var
        else:
            betas.iloc[t] = 0.0

    return betas

# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(
    weights:          pd.DataFrame,
    returns:          pd.DataFrame,
    use_vol_scaling:  bool = True,
    long_only:        bool = False,
    dynamic_weighting:bool = False,
    benchmark_returns:pd.Series = None,
    borrow_cost_annual: float = BASE_BORROW_COST,
    beta_filter: bool = False,
    betas: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Simulate daily portfolio P&L from signal weights and asset returns.

    Parameters
    ----------
    weights : pd.DataFrame
        Target portfolio weights at each rebalancing date.
        From momentum_signal.compute_all_signals().
    returns : pd.DataFrame
        Daily returns matrix. From data_pipeline.load_returns().
    use_vol_scaling : bool
        Whether to apply constant volatility scaling.
        Set to False to reproduce the static baseline.
    long_only : bool
        If True, zero out all short positions. Used to isolate long-leg alpha.
    dynamic_weighting : bool
        If True, apply dynamic scaling from Daniel & Moskowitz (2016) Eq. 6.
        Scales by conditional Sharpe ratio (forecasted return / variance).
        Requires benchmark_returns for the bear market indicator.
        Overrides use_vol_scaling when True.
    benchmark_returns : pd.Series, optional
        Daily market returns for the bear market indicator.
        Required when dynamic_weighting=True.

    Returns
    -------
    pd.DataFrame
        Daily portfolio performance with columns:
            portfolio_return   gross daily return
            net_return         return after transaction costs
            transaction_cost   cost paid (0 on non-rebalancing days)
            turnover           fraction of portfolio traded
            long_return        return of long leg only
            short_return       return of short leg only (0 if long_only)
            vol_scale          scaling factor applied (1.0 if no scaling)
    """
    borrow_cost_daily = borrow_cost_annual / 252
    all_trading_days = returns.index
    common_tickers   = weights.columns.intersection(returns.columns)
    weights          = weights[common_tickers]
    returns          = returns[common_tickers]

    # Align benchmark returns to trading days for dynamic weighting
    market_ret_series = None
    if dynamic_weighting and benchmark_returns is not None:
        market_ret_series = benchmark_returns.reindex(all_trading_days).fillna(0.0)

    current_weights         = pd.Series(0.0, index=common_tickers)
    past_portfolio_returns: list[float] = []
    past_market_returns:    list[float] = []
    results                 = []

    if dynamic_weighting:
        mode = "dynamic-weighting"
    elif long_only:
        mode = "long-only"
    elif use_vol_scaling:
        mode = "vol-scaled"
    else:
        mode = "static"

    logger.info(
        f"Simulating portfolio: {len(weights)} rebalancing dates, "
        f"{len(all_trading_days)} trading days ({mode})"
    )

    for date in all_trading_days:

        is_rebal         = date in weights.index
        transaction_cost = 0.0
        turnover         = 0.0
        vol_scale        = 1.0

        if is_rebal:
            new_weights = weights.loc[date].reindex(common_tickers).fillna(0.0)

            # --- Beta-filtered short leg ---
            if beta_filter and betas is not None:
                beta_today = betas.loc[date].reindex(common_tickers).fillna(0.0)

                short_mask = new_weights < 0

                # Keep only high-beta shorts
                high_beta = beta_today > beta_today.median()

                # Remove low-beta shorts
                new_weights[short_mask & ~high_beta] = 0.0

                total_abs = new_weights.abs().sum()
                if total_abs > 0:
                    new_weights = new_weights / total_abs

            # Long-only: drop all short positions and renormalize
            if long_only:
                new_weights = new_weights.clip(lower=0)
                total = new_weights.sum()
                if total > 0:
                    new_weights = new_weights / total

            # Dynamic weighting overrides constant vol scaling
            if dynamic_weighting:
                vol_scale   = _compute_dynamic_scale(
                    past_portfolio_returns,
                    past_market_returns,
                )
                new_weights = new_weights * vol_scale
            elif use_vol_scaling:
                vol_scale   = _compute_vol_scale(past_portfolio_returns)
                new_weights = new_weights * vol_scale

            turnover         = (new_weights - current_weights).abs().sum() / 2
            transaction_cost = turnover * COST_RATE
            current_weights  = new_weights

        today_returns    = returns.loc[date].reindex(common_tickers).fillna(0.0)
        portfolio_return = (current_weights * today_returns).sum()

        long_mask    = current_weights > 0
        short_mask   = current_weights < 0
        long_return  = (current_weights[long_mask]  * today_returns[long_mask]).sum()
        short_return = (current_weights[short_mask] * today_returns[short_mask]).sum()

        # Borrow cost (short side only)
        short_exposure = current_weights[current_weights < 0].abs().sum()
        borrow_cost    = short_exposure * borrow_cost_daily

        net_return = portfolio_return - transaction_cost - borrow_cost

        # Append before updating weights past returns must not include today
        past_portfolio_returns.append(portfolio_return)
        if market_ret_series is not None:
            past_market_returns.append(float(market_ret_series.loc[date]))

        results.append({
            "date"             : date,
            "portfolio_return" : portfolio_return,
            "net_return"       : net_return,
            "transaction_cost" : transaction_cost,
            "turnover"         : turnover,
            "long_return"      : long_return,
            "short_return"     : short_return,
            "vol_scale"        : vol_scale,
            "borrow_cost": borrow_cost,
            "short_exposure": short_exposure,
        })

        # Drift weights between rebalancings to reflect price moves
        if abs(1 + portfolio_return) > 1e-10:
            current_weights = (
                current_weights * (1 + today_returns)
                / (1 + portfolio_return)
            )

    portfolio_df = pd.DataFrame(results).set_index("date")
    logger.info(f"Simulation complete: {len(portfolio_df)} daily observations")
    return portfolio_df


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def compute_benchmark(returns: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted S&P 500 benchmark mean daily return across all stocks.
    NaN handled naturally (stocks not yet listed contribute nothing).
    """
    benchmark      = returns.mean(axis=1)
    benchmark.name = "benchmark_return"
    return benchmark


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    daily_returns: pd.Series,
    benchmark:     pd.Series = None,
    label:         str = "Strategy",
) -> dict:
    """
    Compute annualized performance metrics.

    Includes Sharpe, Sortino, max drawdown, Calmar, monthly hit rate,
    and alpha/beta versus the benchmark.

    Risk-free rate is set to zero a reasonable simplification
    for a long-short dollar-neutral strategy.
    """
    r = daily_returns.dropna()

    n_years       = len(r) / 252
    total_return  = (1 + r).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_vol    = r.std() * np.sqrt(252)
    sharpe        = annual_return / annual_vol if annual_vol > 0 else 0

    downside_vol = r[r < 0].std() * np.sqrt(252)
    sortino      = annual_return / downside_vol if downside_vol > 0 else 0

    cumulative   = (1 + r).cumprod()
    drawdown     = (cumulative - cumulative.cummax()) / cumulative.cummax()
    max_drawdown = drawdown.min()
    calmar       = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    monthly_returns = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    hit_rate        = (monthly_returns > 0).mean()

    alpha = beta = alpha_tstat = None
    if benchmark is not None:
        b      = benchmark.reindex(r.index).dropna()
        common = r.reindex(b.index).dropna()
        b      = b.reindex(common.index)

        if len(common) > 60:
            X      = np.column_stack([np.ones(len(b)), b.values])
            y      = common.values
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha_daily = coeffs[0]
            beta        = coeffs[1]
            alpha       = alpha_daily * 252

            n, k = len(y), 2
            if len(residuals) > 0:
                resid_std   = np.sqrt(residuals[0] / (n - k))
                alpha_se    = resid_std * np.sqrt(np.linalg.inv(X.T @ X)[0, 0])
                alpha_tstat = alpha_daily / alpha_se if alpha_se > 0 else 0

    metrics = {
        "label"        : label,
        "annual_return": round(annual_return, 4),
        "annual_vol"   : round(annual_vol,    4),
        "sharpe"       : round(sharpe,         2),
        "sortino"      : round(sortino,         2),
        "max_drawdown" : round(max_drawdown,   4),
        "calmar"       : round(calmar,          2),
        "hit_rate"     : round(hit_rate,        4),
        "total_return" : round(total_return,   4),
        "n_years"      : round(n_years,         1),
        "alpha"        : round(alpha,       4) if alpha       is not None else None,
        "beta"         : round(beta,        4) if beta        is not None else None,
        "alpha_tstat"  : round(alpha_tstat, 2) if alpha_tstat is not None else None,
    }

    logger.info(f"\n{'='*45}")
    logger.info(f"  {label}")
    logger.info(f"{'='*45}")
    logger.info(f"  Annual return    : {annual_return:+.2%}")
    logger.info(f"  Annual vol       : {annual_vol:.2%}")
    logger.info(f"  Sharpe ratio     : {sharpe:.2f}")
    logger.info(f"  Sortino ratio    : {sortino:.2f}")
    logger.info(f"  Max drawdown     : {max_drawdown:.2%}")
    logger.info(f"  Calmar ratio     : {calmar:.2f}")
    logger.info(f"  Monthly hit rate : {hit_rate:.1%}")
    if alpha is not None:
        logger.info(f"  Alpha (annual)   : {alpha:+.2%} (t={alpha_tstat:.2f})")
        logger.info(f"  Beta             : {beta:.3f}")
    logger.info(f"{'='*45}")

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validation(
    weights:   pd.DataFrame,
    returns:   pd.DataFrame,
    benchmark: pd.Series,
    n_folds:   int = N_FOLDS,
) -> pd.DataFrame:
    """
    Split the backtest into N sequential folds and report metrics per fold.

    Folds are strictly sequential no data from fold N is visible
    when computing fold N-1. A robust strategy performs consistently
    across all folds rather than concentrating returns in one period.
    """
    logger.info(f"Running walk-forward validation ({n_folds} folds)...")

    portfolio   = simulate_portfolio(weights, returns)
    net_returns = portfolio["net_return"]
    all_dates   = net_returns.index
    fold_size   = len(all_dates) // n_folds
    fold_results = []

    for fold in range(n_folds):
        start_idx    = fold * fold_size
        end_idx      = (fold + 1) * fold_size if fold < n_folds - 1 else len(all_dates)
        fold_dates   = all_dates[start_idx:end_idx]
        fold_returns = net_returns.loc[fold_dates]
        fold_bench   = benchmark.reindex(fold_dates)

        label   = f"Fold {fold+1} ({fold_dates[0].strftime('%Y-%m')} → {fold_dates[-1].strftime('%Y-%m')})"
        metrics = compute_metrics(fold_returns, fold_bench, label)
        metrics.update({"fold": fold + 1, "start_date": fold_dates[0].date(), "end_date": fold_dates[-1].date()})
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def analyze_regimes(
    portfolio:  pd.DataFrame,
    benchmark:  pd.Series,
) -> pd.DataFrame:
    """
    Report performance across three distinct market regimes:
      - Bull market (2010-2019): low vol, steady uptrend
      - COVID crash/recovery (2020-2021): sharp crash then violent rebound
      - Rate hike period (2022-2023): high inflation, sector rotation

    The COVID period is where momentum crashes are expected per
    Daniel & Moskowitz (2016) bear market followed by sharp rebound.
    """
    regimes = [
        ("Bull market",          "2010-01-01", "2019-12-31"),
        ("COVID crash/recovery", "2020-01-01", "2021-12-31"),
        ("Rate hike period",     "2022-01-01", "2023-12-31"),
    ]
    net_returns    = portfolio["net_return"]
    regime_results = []

    for name, start, end in regimes:
        mask         = (net_returns.index >= start) & (net_returns.index <= end)
        period_ret   = net_returns[mask]
        period_bench = benchmark.reindex(period_ret.index)

        if len(period_ret) < 20:
            continue

        metrics           = compute_metrics(period_ret, period_bench, name)
        metrics["regime"] = name
        regime_results.append(metrics)

    return pd.DataFrame(regime_results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
 
    os.makedirs("results", exist_ok=True)
 
    # --- Load data ---------------------------------------------------------
    logger.info("=== Loading data ===")
    returns    = load_returns()
    sector_map = load_sector_map()
 
    signals_path = "data/processed/signals.parquet"
    if os.path.exists(signals_path):
        logger.info("Loading precomputed signals...")
        weights = pd.read_parquet(signals_path)
    else:
        logger.info("Computing signals...")
        rebalancing_dates = get_rebalancing_dates(START_DATE, END_DATE)
        weights           = compute_all_signals(returns, rebalancing_dates, sector_map)
        weights.to_parquet(signals_path)
 
    benchmark      = compute_benchmark(returns)
    market_returns = load_market_returns()
 
    betas = compute_rolling_beta(returns, benchmark)
 
    # --- Baseline (no vol scaling) -----------------------------------------
    logger.info("=== Baseline backtest (static WML) ===")
    portfolio_static = simulate_portfolio(weights, returns, use_vol_scaling=False)
    metrics_static   = compute_metrics(
        portfolio_static["net_return"], benchmark, "Static WML (baseline)"
    )
 
    # --- Vol-scaled long-short --------------------------------------------
    logger.info("=== Vol-scaled long-short ===")
    portfolio_scaled = simulate_portfolio(weights, returns, use_vol_scaling=True)
    portfolio_scaled.to_parquet(PORTFOLIO_PATH)
    metrics_scaled   = compute_metrics(
        portfolio_scaled["net_return"], benchmark, "Vol-Scaled WML"
    )
 
 
    # --- Beta Filtering on vol scal --------------------------------
    logger.info("=== Beta-filtered short strategy on vol scal ===")
 
    portfolio_beta = simulate_portfolio(
        weights,
        returns,
        use_vol_scaling=True,
        borrow_cost_annual=BASE_BORROW_COST,
        beta_filter=True,
        betas=betas,
    )
 
    metrics_beta = compute_metrics(
        portfolio_beta["net_return"],
        benchmark,
        "Beta-Filtered Shorts"
    )
 
 
    # --- Dynamic weighting (Daniel & Moskowitz 2016, Eq. 6) ---------------
    # Scales by conditional Sharpe ratio using rolling OLS forecast
    # of expected return and trailing realized variance.
    # All estimates are strictly out-of-sample (expanding window).
    logger.info("=== Dynamic weighting backtest ===")
    portfolio_dynamic = simulate_portfolio(
        weights, returns,
        dynamic_weighting=True,
        benchmark_returns=market_returns,
    )
    metrics_dynamic   = compute_metrics(
        portfolio_dynamic["net_return"], benchmark, "Dynamic Weighting (D&M 2016)"
    )

    # --- Beta-filtered Dynamic weighting ---
    logger.info("=== Beta-filtered Dynamic weighting ===")
    portfolio_beta_dynamic = simulate_portfolio(
        weights,
        returns,
        dynamic_weighting=True,
        benchmark_returns=market_returns,
        beta_filter=True,
        betas=betas,
    )

    metrics_beta_dynamic = compute_metrics(
        portfolio_beta_dynamic["net_return"],
        benchmark,
        "Beta-Filtered Dynamic"
    )

    # --- Long-only (vol-scaled, no shorts) --------------------------------
    # Isolates the long-leg alpha to separate signal quality from
    # the short-side underperformance documented in the decomposition.
    logger.info("=== Long-only backtest ===")
    portfolio_long   = simulate_portfolio(weights, returns, use_vol_scaling=True, long_only=True)
    metrics_long     = compute_metrics(
        portfolio_long["net_return"], benchmark, "Long-Only (top quintile)"
    )
 
    # --- Benchmark --------------------------------------------------------
    bench_metrics = compute_metrics(benchmark, label="Equal-Weight Benchmark")
 
    # --- Walk-forward validation (vol-scaled long-short) ------------------
    logger.info("=== Walk-forward validation ===")
    wf_results = walk_forward_validation(weights, returns, benchmark)
 
    # --- Regime analysis (vol-scaled long-short) --------------------------
    logger.info("=== Regime analysis ===")
    regime_results = analyze_regimes(portfolio_scaled, benchmark)
 
    # --- Borrow cost sensitivity ------------------------------------------
    logger.info("=== Borrow Cost Sensitivity Analysis ===")
 
    borrow_costs = [0.0, 0.01, 0.03, 0.05]
    sensitivity_results = []
 
    for bc in borrow_costs:
        logger.info(f"Running backtest with borrow cost = {bc:.1%} on vol scal")
 
        portfolio_bc = simulate_portfolio(
            weights,
            returns,
            use_vol_scaling=True,
            borrow_cost_annual=bc,   
        )
 
        metrics_bc = compute_metrics(
            portfolio_bc["net_return"],
            benchmark,
            label=f"Borrow {bc:.1%}"
        )
 
        sensitivity_results.append({
            "borrow_cost": bc,
            "sharpe": metrics_bc["sharpe"],
            "annual_return": metrics_bc["annual_return"],
            "max_drawdown": metrics_bc["max_drawdown"],
        })
 
    # --- Print sensitivity table ------------------------------------------
    print("\n--- Borrow Cost Sensitivity ---")
    print(f"{'Borrow Cost':<15} {'Sharpe':>10} {'Return':>12} {'Drawdown':>12}")
    print("-" * 55)
 
    for row in sensitivity_results:
        print(
            f"{row['borrow_cost']:<15.1%} "
            f"{row['sharpe']:>10.2f} "
            f"{row['annual_return']:>12.2%} "
            f"{row['max_drawdown']:>12.2%}"
        )
 
    # --- Save all metrics -------------------------------------------------
    pd.DataFrame([metrics_static, metrics_scaled, metrics_beta, metrics_dynamic, metrics_beta_dynamic, metrics_long, bench_metrics]).to_csv(
        METRICS_PATH, index=False
    )
 
    # --- Summary ----------------------------------------------------------
    print("\n========== Backtest Summary ==========")
    net_scaled = portfolio_scaled["net_return"]
    print(f"Period       : {net_scaled.index[0].date()} → {net_scaled.index[-1].date()}")
    print(f"Trading days : {len(net_scaled)}")
 
    print(f"\n--- Results progression ---")
    print(f"{'Metric':<22} {'Static L/S':>12} {'Vol-Scaled':>12} {'Beta-Filt':>12} {'Dynamic':>12} {'Beta-Filt Dyn':>14} {'Long-Only':>12} {'Benchmark':>12}")
    print("-" * 112)
    for key in ["annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar"]:
        vals = [metrics_static, metrics_scaled, metrics_beta, metrics_dynamic, metrics_beta_dynamic, metrics_long, bench_metrics]
        row  = []
        for m in vals:
            v = m.get(key)
            if isinstance(v, float):
                row.append(f"{v:+.2%}" if key in ["annual_return", "annual_vol", "max_drawdown"] else f"{v:.2f}")
            else:
                row.append("N/A")
        print(f"  {key:<20} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>14} {row[5]:>12} {row[6]:>12}")
        
    print(f"\n  Alpha - vol-scaled  : {metrics_scaled.get('alpha', 0):+.2%} (t={metrics_scaled.get('alpha_tstat', 0):.2f})")
    print(f"  Alpha - beta-filt-vol-scaled   : {metrics_beta.get('alpha', 0):+.2%} (t={metrics_beta.get('alpha_tstat', 0):.2f})")
    print(f"  Alpha - dynamic     : {metrics_dynamic.get('alpha', 0):+.2%} (t={metrics_dynamic.get('alpha_tstat', 0):.2f})")
    print(f"  Alpha - beta-filt-dynamic     : {metrics_beta_dynamic.get('alpha', 0):+.2%} (t={metrics_beta_dynamic.get('alpha_tstat', 0):.2f})")
    print(f"  Alpha - long-only   : {metrics_long.get('alpha', 0):+.2%} (t={metrics_long.get('alpha_tstat', 0):.2f})")
    print(f"  Beta  - vol-scaled  : {metrics_scaled.get('beta', 0):.3f}")
    print(f"  Beta  - beta-filt-vol-scal   : {metrics_beta.get('beta', 0):.3f}")
    print(f"  Beta  - dynamic     : {metrics_dynamic.get('beta', 0):.3f}")
    print(f"  Beta  - beta-filt-dynamic     : {metrics_beta_dynamic.get('beta', 0):.3f}")
    print(f"  Beta  - long-only   : {metrics_long.get('beta', 0):.3f}")
 
    print(f"\n--- Walk-Forward Validation (vol-scaled long-short) ---")
    for _, row in wf_results.iterrows():
        print(f"  {row['label']:<42} Sharpe: {row['sharpe']:+.2f}  Return: {row['annual_return']:+.2%}")
 
    print(f"\n--- Regime Analysis (vol-scaled) ---")
    for _, row in regime_results.iterrows():
        print(f"  {row['regime']:<25} Sharpe: {row['sharpe']:+.2f}  Return: {row['annual_return']:+.2%}")
 
    avg_scale    = portfolio_scaled[portfolio_scaled["vol_scale"] != 1.0]["vol_scale"].mean()
    avg_turnover = portfolio_scaled[portfolio_scaled["turnover"] > 0]["turnover"].mean()
    total_costs  = portfolio_scaled["transaction_cost"].sum()
    print(f"\n--- Vol Scaling Stats ---")
    print(f"  Avg scale factor     : {avg_scale:.2f}x")
    print(f"  Avg monthly turnover : {avg_turnover:.1%}")
    print(f"  Annualized cost drag : {total_costs / len(net_scaled) * 252:.2%}")
    print("======================================")