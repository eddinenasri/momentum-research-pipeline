
# Cross-Sectional Momentum 

**Author:** Eddine Nasri · [eddinenasri@gmail.com](mailto:eddinenasri@gmail.com)

> **Full methodology, results, and performance attribution:**
> [`notebooks/research.ipynb`](notebooks/research.ipynb) · [`notebook.html`](notebook.html) *(pre-rendered, no setup required)*

---

## Overview

This project builds a complete quantitative research pipeline for a **cross-sectional momentum strategy** on the S&P 500 (2010–2023). It covers every layer of the research process from raw data acquisition to rigorous backtesting all from scracth and literature-driven improvement without relying on any pre-built backtesting library.

The goal is not to produce a profitable strategy, but to build a complete infrastructure that allows implementing a strategy from an academic paper, managing and cleaning market data, and constructing a backtest entirely from scratch to evaluate it. Additional research papers are also incorporated to refine and enhance the strategy, ensuring a rigorous and fully understandable assessment of its performance.

Concretely, this means: constructing a point-in-time survivorship-bias-free universe, implementing the **12-1 momentum signal** of Jegadeesh & Titman (1993) with the institutional refinements from Asness, Moskowitz & Pedersen (2013) Sharpe scaling, sector neutralisation, inverse-volatility sizing running a custom backtesting engine with realistic transaction costs and weight drift, and applying the volatility-targeting crash-protection framework of Daniel & Moskowitz (2016). Performance is then decomposed across regimes, walk-forward folds, and long/short legs to isolate exactly where alpha is generated and where it is not.

---

## Research Notebooks

The full analysis signal diagnostics, performance attribution, walk-forward validation across 5 sequential folds, regime decomposition, and long/short leg breakdown is documented in the notebooks. **Open these first.**

| Notebook | Format | How to use |
|---|---|---|
| [`notebook.html`](notebook.html) | Pre-rendered HTML | Open directly in a browser no setup required |
| [`notebooks/research.ipynb`](notebooks/research.ipynb) | Jupyter notebook | Fork the repo, run the pipeline, explore interactively |

---

## Methodology

### 1. Data Infrastructure

Building a backtestable dataset is the first and most underestimated engineering problem in systematic research. Several sources of bias must be eliminated before any strategy code is written.

**Point-in-time universe construction** S&P 500 composition sourced from [fja05680/sp500](https://github.com/fja05680/sp500), providing daily historical membership since 1996. At each rebalancing date the model sees only stocks that were constituents *on that date*. Naively using today's index introduces survivorship bias inflating backtest performance by silently excluding every historical loser that was since delisted or removed.

**Price data pipeline** Adjusted close prices downloaded for all 794 unique tickers ever in the index over the backtest window, not just the current 500. Downloaded in batches with graceful failure handling for delisted tickers.

**Data cleaning** Forward-fill capped at 3 days (market holidays, brief suspensions); stocks with under 252 trading days excluded (insufficient history for a 12-month signal); daily returns beyond ±50% winsorised to cap data errors without distorting the cross-sectional return distribution; all series reindexed to a common trading calendar. Stored in Parquet ~10× faster to read and ~5× smaller than CSV for numerical matrices of this size.

---

### 2. Signal Construction

The signal is built on the **12-1 momentum** specification: compounded return over the past 12 months, skipping the most recent month. The one-month skip eliminates short-term reversal contamination a documented microstructure artefact first identified by Jegadeesh (1990) where recent sharp moves are more likely to mean-revert than persist.

Four refinements are applied on top of the raw signal:

**Sharpe scaling** Raw momentum is biased toward high-volatility names. A stock returning +30% with 40% annualised volatility has weaker signal quality than one returning +20% with 10% volatility. Dividing by realised volatility over the same window produces a risk-adjusted signal:

$$\text{signal}_i = \frac{r^{12\text{-}1}_i}{\hat{\sigma}_i}$$

**Sector neutralisation** Without it, the long book concentrates in whichever sector had the strongest macro tailwind not a diversified momentum portfolio, but a levered sector bet. Subtracting the sector median within each GICS group transforms the signal from "which stocks have the strongest absolute momentum?" to "which stocks outperformed their sector peers?", producing a diversified book across all sectors.

**Inverse-volatility position sizing** Equal notional weighting lets high-volatility names dominate portfolio risk. Weighting by $1/\hat{\sigma}_i$ ensures each position contributes approximately equal marginal risk to the portfolio.

---

### 3. Backtesting Engine

The simulation is implemented entirely in Python without any backtesting library. This is a deliberate design choice: pre-built frameworks abstract away the mechanics that matter most for an honest evaluation.

**Execution model** Signal computed at close of date $T$; positions entered at open of date $T+1$. Using the same-day close for both would introduce look-ahead bias.

**Weight drift** Between rebalancing dates, positions drift freely as prices move. The engine updates weights daily to reflect realised price changes:

$$w_{i,t+1} = w_{i,t} \cdot \frac{1 + r_{i,t}}{1 + r_{p,t}}$$

This avoids the implicit assumption made by most naive backtesters that the portfolio magically rebalances to target weights each day.

**Transaction costs** 10 bps one-way applied to actual turnover only. Turnover analysis showed ~60% monthly turnover is structural (index additions/removals), not signal-driven only 0.3 positions per month flip direction on the signal itself.

**Walk-forward validation** 5 sequential folds with zero data leakage, each evaluated independently to distinguish consistent performance from regime-specific luck.

---

### 4. Crash Protection - Daniel & Moskowitz (2016)

After identifying negative performance in the full long-short, the research turned to the Daniel & Moskowitz (2016) *Momentum Crashes* framework. The paper documents that momentum drawdowns are **predictable** and concentrated in "panic states": environments where the market has already fallen for 24 months and current volatility is elevated. In these conditions, the short leg beaten-down, distressed companies behaves like a portfolio of call options with asymmetric payoffs. The loser decile rose 163% during March–May 2009.

**Implementation** Constant volatility scaling targets 10% annualised portfolio volatility at each rebalancing date:

$$\text{scale}_t = \frac{\sigma_{\text{target}}}{\hat{\sigma}_{t-1}^{WML}}, \quad \text{capped at } 2\times$$

where $\hat{\sigma}_{t-1}^{WML}$ is the realised volatility of daily WML returns over the past 126 trading days. When volatility is elevated the regime that precedes crashes exposure is automatically reduced before the crash materialises.

---

## Repository Structure

```
├── src/
│   ├── universe.py          # Point-in-time S&P 500 membership
│   ├── data_pipeline.py     # Download, clean, compute returns
│   ├── momentum_signal.py   # Signal construction, portfolio weights
│   └── backtest.py          # Simulation, metrics, walk-forward validation
├── data/
│   ├── universe/            # Historical constituents CSV, GICS sector map
│   ├── raw/                 # Raw prices, market returns
│   └── processed/           # Clean prices, returns, signals (Parquet)
├── results/
│   └── metrics.csv
├── notebooks/
│   └── research.ipynb
└── notebook.html            # Pre-rendered research notebook
```

---

## Quickstart

```bash
git clone https://github.com/eddinenasri/momentum-research-pipeline.git
cd momentum-research-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download historical S&P 500 universe
curl -L "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv" \
     -o "data/universe/sp500_historical_components.csv"

# Run pipeline (10–15 min end-to-end)
python src/data_pipeline.py      # Download and clean ~794 tickers
python src/momentum_signal.py    # Compute signals for 144 rebalancing dates
python src/backtest.py           # Run all strategy variants

# Research notebook
jupyter notebook notebooks/notebook.ipynb
```

---

## References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.* Journal of Finance, 48(1), 65–91.
- Daniel, K. & Moskowitz, T. (2016). *Momentum Crashes.* Journal of Financial Economics, 122(2), 221–247.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere.* Journal of Finance, 68(3), 929–985.
- Moskowitz, T. & Grinblatt, M. (1999). *Do Industries Explain Momentum?* Journal of Finance, 54(4), 1249–1290.
- McLean, R. & Pontiff, J. (2016). *Does Academic Research Destroy Stock Return Predictability?* Journal of Finance, 71(1), 5–32.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Grinold, R. & Kahn, R. (1999). *Active Portfolio Management.* McGraw-Hill.

---

*For results, performance attribution, and findings open the notebook.*