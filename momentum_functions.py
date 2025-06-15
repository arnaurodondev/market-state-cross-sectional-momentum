import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthEnd, MonthBegin
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy import stats
from scipy.linalg import eigvals

"""
================================================================================
MARKET-STATE DEPENDENT MOMENTUM STRATEGY: IMPLEMENTATION AND ANALYSIS
================================================================================

This module implements a conditional momentum-reversal trading strategy that
dynamically switches between momentum and contrarian approaches based on 
prevailing market conditions, as described in:

"Winners & Losers in Motion: A Market-State Signal that Reverses Roles for Robust
Returns (1940-2024)" (Rodon Comas, 2025)

Key Features:
- Dynamic regime switching based on 24-month market state classification
- Transaction cost modeling with bid-ask spreads and commissions
- Overlapping portfolio construction following Jegadeesh & Titman (1993)
- HAC-adjusted statistical inference accounting for both holding period and market state lag
- Comprehensive factor model analysis (FF5 + momentum + reversal)
- HAC lag structure now accounts for market state lookback period
- Lag = K + market_state_lag - 1 (e.g., K=3, market_state_lag=24 → lag=26)
- This properly captures autocorrelation from both overlapping portfolios and state persistence

Author: Arnau Rodon Comas
Date: 25/05/2025
================================================================================
"""

def compute_market_state(crsp_index: pd.DataFrame, 
                        lags: int = 24, 
                        column: str = 'vwretd', 
                        date_column: str = 'date') -> pd.DataFrame:
    """
    Compute market states based on cumulative market returns over prior periods.
    
    This function implements the market state identification methodology from
    Cooper, Gutierrez, and Hameed (2004), classifying market conditions as
    UP (1) or DOWN (-1) based on cumulative returns over the specified lag period.
    
    The choice of lag period is critical:
    - 12 months: More responsive but potentially noisy (77 state changes 1940-2023)
    - 24 months: Balanced approach, our primary specification (47 state changes)
    - 36 months: More stable but less responsive (18 state changes)
    
    Parameters
    ----------
    crsp_index : pd.DataFrame
        CRSP market index data containing date and return columns
        Expected columns: date, vwretd (value-weighted), ewretd (equal-weighted)
    lags : int, default=24
        Lookback period in months for cumulative return calculation
        Literature suggests 12-36 months, with 24 as optimal balance
    column : str, default='vwretd'
        Return column to use ('vwretd' for value-weighted, 'ewretd' for equal-weighted)
        Value-weighted typically preferred for market state classification
    date_column : str, default='date'
        Name of the date column in the DataFrame
        
    Returns
    -------
    pd.DataFrame
        Original data with additional columns:
        - return_plus_one: 1 + return (for compounding)
        - expected_cumulative_return: Cumulative return over lag period
        - market_state: 1 for UP (positive cumulative), -1 for DOWN (negative)
        
    Notes
    -----
    The function uses a multiplicative (geometric) approach for calculating
    cumulative returns, which properly accounts for compounding effects:
    
    Cumulative Return = ∏(1 + r_t) - 1
    
    This is more accurate than simple summation for multi-period returns.
    
    References
    ----------
    Cooper, M.J., Gutierrez Jr, R.C., & Hameed, A. (2004). Market states and momentum.
    The Journal of Finance, 59(3), 1345-1365.
    """
    
    # Create a copy to avoid modifying the original data
    crsp = crsp_index.copy()
    
    # Remove any unnamed index columns from data loading
    crsp = crsp.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Ensure date column is datetime type for proper time series operations
    crsp['date'] = pd.to_datetime(crsp[date_column])
    
    # Calculate (1 + return) for proper geometric compounding
    # This handles the multiplicative nature of returns correctly
    crsp['return_plus_one'] = 1 + crsp[column]
    
    # Calculate cumulative return using rolling window with multiplicative approach
    # shift(1) ensures we use lagged values (avoiding look-ahead bias)
    crsp['expected_cumulative_return'] = (
        crsp['return_plus_one']
        .rolling(window=lags, min_periods=lags)  # Require full window
        .apply(np.prod, raw=True)                # Product of (1+r)
        .shift(1) - 1                            # Lag by 1 month and convert to return
    )
    
    # Determine market state based on cumulative return sign
    # UP market (1): Positive cumulative return over lag period
    # DOWN market (-1): Negative cumulative return over lag period
    crsp['market_state'] = np.where(
        crsp['expected_cumulative_return'] > 0, 
        1,   # UP state
        -1   # DOWN state
    )
    
    # Remove observations without valid market state (initial lag periods)
    crsp = crsp.dropna(subset=['market_state']).reset_index(drop=True)
    
    return crsp


def run_momentum_strategy(crsp_data, crsp_index_data, comm_map, 
                         J=12, K=3, LIQ_CUTOFF=0.85, MKT_CAP_CUTOFF=1.0,
                         market_state_lag=24, market_state_column='vwretd',
                         transactional_framework=True, equal_comparision=False,
                         adapt_state_switch=True, close_positions=False, 
                         amount_deciles=10, start_date='1993-01-01'):
    """
    Implements a momentum strategy with market capitalization and liquidity constraints.
    
    This function executes a momentum-based trading strategy that forms portfolios based on
    past returns (formation period) and holds them for a specified period (holding period).
    The strategy incorporates transaction costs, market state adaptations, and various
    filtering criteria following academic best practices.
    
    Strategy Logic:
    1. Rank stocks into deciles based on J-month cumulative returns
    2. Apply size and liquidity filters to ensure tradability
    3. Form long-short portfolios (long winners, short losers)
    4. Adapt positions based on market state (momentum vs reversal)
    5. Account for transaction costs (spreads and commissions)
    
    Parameters
    ----------
    crsp_data : pd.DataFrame
        CRSP stock data with required columns:
        - permno: Permanent security identifier
        - date: Trading date
        - ret: Simple return
        - logret: Log return (for accurate compounding)
        - half_spread: Half bid-ask spread (transaction cost component)
        - mktcap: Market capitalization (lagged to avoid look-ahead bias)
        
    crsp_index_data : pd.DataFrame
        CRSP index data for market state calculations
        
    comm_map : dict
        Dictionary mapping date strings (YYYY-MM) to commission rates
        Reflects historical brokerage costs evolution
        
    J : int, default=12
        Formation period length in months (lookback for momentum calculation)
        Literature suggests 3-12 months, with 12 being most robust
        
    K : int, default=3
        Holding period length in months
        Shorter periods (1-3) capture momentum, longer periods may include reversal
        
    LIQ_CUTOFF : float, default=0.85
        Liquidity cutoff percentile (keeps most liquid 85% of stocks)
        Excludes illiquid stocks with high transaction costs
        
    MKT_CAP_CUTOFF : float, default=1.0
        Market cap cutoff percentile (1.0 means no size restriction)
        Can be lowered to focus on specific size segments
        
    market_state_lag : int, default=24
        Lookback period for market state classification (months)
        24 months balances responsiveness with stability
        
    market_state_column : str, default='vwretd'
        Index return series for state classification
        'vwretd': value-weighted, 'ewretd': equal-weighted
        
    transactional_framework : bool, default=True
        Apply realistic trading constraints:
        - Transaction costs (bid-ask spreads + commissions)
        - Date filters ensuring tradability
        - Exclusion of dot-com crash period if applicable
        
    equal_comparision : bool, default=False
        Use equal-weighted portfolio construction for comparison
        
    adapt_state_switch : bool, default=True
        Allow strategy to adapt when market state changes mid-holding period
        Critical for capturing regime-dependent returns
        
    close_positions : bool, default=False
        True: Close positions on state switch (conservative)
        False: Reverse positions on state switch (aggressive)
        
    amount_deciles : int, default=10
        Number of quantiles for momentum sorting
        Standard is 10 (deciles), but can use 5 (quintiles) for smaller samples
        
    start_date : str or pd.Timestamp, default='1993-01-01'
        First tradable date (after formation period buffer)
        Pre-1993 may have data quality issues
        
    Returns
    -------
    portfolio : pd.DataFrame
        Detailed portfolio holdings and returns with columns:
        - Basic identifiers (permno, date, form_date)
        - Returns for different strategy variants (RET_S0, S1, S2, S3)
        - Market state indicators
        - Transaction cost components
        
    state_map : dict
        Mapping of date strings to market states for reference
        
    Notes
    -----
    Strategy Variants:
    - S0: Raw momentum without costs (benchmark)
    - S1: Market state reversal without costs (reversal in DOWN markets)
    - S2: Standard momentum with transaction costs
    - S3: Full conditional strategy with costs and state adaptation
    
    The dot-com crash exclusion (Oct 2000 - Apr 2001) follows literature
    documenting extreme momentum crashes during this period.
    """
    
    # ========================================================================
    # INITIALIZATION AND DATA PREPARATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"IMPLEMENTING MOMENTUM STRATEGY WITH MARKET CAP CONSTRAINTS")
    print(f"Parameters: J={J}, K={K}, LIQ={LIQ_CUTOFF}, MKT_CAP={MKT_CAP_CUTOFF}")
    
    # ------------------------------------------------------------------------
    # Apply date filters for transactional framework
    # ------------------------------------------------------------------------
    if transactional_framework:
        # Ensure we have K months of data before first tradable date
        # This allows momentum deciles to be formed for the first trading month
        start_date  = pd.to_datetime(start_date)
        filter_date = start_date - pd.DateOffset(months=K)
        print(f"Transactional Framework: Filtering data from "
              f"{filter_date.date()} (start_date {start_date.date()} minus {K} months)")

        # Convert date column to datetime for proper comparison
        crsp_data['date'] = pd.to_datetime(crsp_data['date'])
        
        # Keep only observations from filter_date onward
        crsp_data = crsp_data[crsp_data['date'] >= filter_date].copy()

        # --------------------------------------------------------------------
        # Conditional exclusion of the dot-com crash (Oct-2000 → Apr-2001)
        # --------------------------------------------------------------------
        # Literature documents severe momentum crashes during this period
        # Exclusion improves out-of-sample strategy robustness
        dotcom_cutoff = pd.Timestamp("2001-04-01")
        n_before_exclusion = len(crsp_data)

        if start_date < dotcom_cutoff:
            # Define months to exclude (inclusive)
            excluded_months = pd.date_range("2000-10-01", "2001-04-01", freq="MS")

            # Create temporary month column for filtering
            crsp_data["temp_month"] = (
                crsp_data["date"]
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            
            # Remove observations from excluded months
            crsp_data = (
                crsp_data[~crsp_data["temp_month"].isin(excluded_months)]
                .drop("temp_month", axis=1)
            )
            n_after_exclusion = len(crsp_data)

            print("Excluded Oct 2000 – Apr 2001 (dot-com crash period)")
            print(f"Removed {n_before_exclusion - n_after_exclusion:,} observations from excluded months")
        else:
            print("Dot-com crash exclusion skipped (start_date ≥ 2001-04-01)")

    print("=" * 80)

    # Copy only necessary columns to optimize memory usage
    needed_cols = ['permno', 'date', 'ret', 'logret', 'half_spread', 'mktcap']
    data = crsp_data[needed_cols].copy()
    data['date'] = pd.to_datetime(data['date'])

    # ========================================================================
    # STEP 1: PREPARE MARKET CAPITALIZATION DATA
    # ========================================================================
    print("\n[1/8] Merging market capitalization data...")
    
    # Sort for efficient groupby operations
    data = data.sort_values(['permno', 'date'])
    grouped = data.groupby('permno', sort=False)
    
    # Lag market cap by one month to avoid look-ahead bias
    # We use last month's market cap for this month's portfolio formation
    data['mktcap'] = grouped['mktcap'].shift(1)
    n_with_mktcap = data['mktcap'].notna().sum()
    print(f"   ✓ Market cap data available for {n_with_mktcap:,} observations")

    # ========================================================================
    # STEP 2: CALCULATE HOLDING PERIOD SPREADS
    # ========================================================================
    print(f"\n[2/8] Computing holding period ({K}-month) spreads...")
    
    # Transaction costs occur at three points:
    # 1. current_spread: For information/filtering at current time
    # 2. spread_open: When opening position (t+1)
    # 3. spread_close: When closing position (t+K+1)
    data['current_spread'] = data['half_spread']
    data['spread_open'] = grouped['half_spread'].shift(1)      # Next month's spread
    data['spread_close'] = grouped['spread_open'].shift(-K)    # K months later

    # ========================================================================
    # STEP 3: CALCULATE FORMATION PERIOD RETURNS
    # ========================================================================
    print(f"\n[3/8] Calculating {J}-month formation period returns...")
    
    # Use log returns for accurate multi-period compounding
    # Sum of log returns = log of product of gross returns
    data['sum_logret'] = grouped['logret'].transform(
        lambda x: x.shift(1).rolling(J, min_periods=J).sum()
    )
    # Convert back to simple cumulative return
    data['cumret'] = np.expm1(data['sum_logret'])  # exp(x) - 1

    # ========================================================================
    # STEP 4: APPLY FILTERS (MARKET CAP AND LIQUIDITY)
    # ========================================================================
    print(f"\n[4/8] Applying market cap (≤{MKT_CAP_CUTOFF*100:.0f}%) and liquidity filters...")

    if transactional_framework:
        # Require complete data for all transaction cost components
        required_fields = ['cumret', 'spread_open', 'spread_close', 'mktcap', 'current_spread']
        mask = data[required_fields].notna().all(axis=1)
        signal = data.loc[mask, ['permno', 'date', 'cumret', 'spread_open', 
                                 'spread_close', 'mktcap', 'current_spread']].copy()
        
        n_before_filters = len(signal)

        # Apply market cap filter
        # Focus on smaller stocks if MKT_CAP_CUTOFF < 1.0
        date_groups = signal.groupby('date', sort=False)
        signal['mkt_cap_rank'] = date_groups['mktcap'].rank(pct=True)
        mktcap_mask = signal['mkt_cap_rank'] >= 1 - MKT_CAP_CUTOFF
        signal = signal[mktcap_mask].copy()
        n_after_mktcap = len(signal)

        # Apply liquidity filter based on spreads
        # Lower spreads indicate higher liquidity
        date_groups = signal.groupby('date', sort=False)
        signal['spread_rank'] = date_groups['spread_open'].rank(pct=True)
        liq_mask = signal['spread_rank'] <= LIQ_CUTOFF
        signal = signal[liq_mask].copy()
        n_after_both = len(signal)

        print(f"   ✓ Market cap filter: {n_before_filters:,} → {n_after_mktcap:,} " + 
              f"({(1-n_after_mktcap/n_before_filters)*100:.1f}% removed)")
        print(f"   ✓ Liquidity filter: {n_after_mktcap:,} → {n_after_both:,} " + 
              f"({(1-n_after_both/n_after_mktcap)*100:.1f}% removed)")
    else:
        # Without transactional framework, only require cumulative returns
        mask = data['cumret'].notna()
        signal = data.loc[mask, ['permno', 'date', 'cumret', 'spread_open', 
                                'spread_close', 'mktcap', 'current_spread']].copy()

    # ========================================================================
    # STEP 5: FORM MOMENTUM DECILES
    # ========================================================================
    print(f"\n[5/8] Forming momentum deciles within size segment...")

    # Sort stocks into momentum deciles based on formation period returns
    # Decile 1: Losers (lowest past returns)
    # Decile 10: Winners (highest past returns)
    signal['momr'] = signal.groupby('date', sort=False)['cumret'].transform(
        lambda x: pd.qcut(x, amount_deciles, labels=False, duplicates='drop') + 1
    )

    # Display decile distribution for data quality check
    decile_counts = signal.groupby(['date', 'momr'], sort=False).size().groupby('momr').mean()
    print("   ✓ Average stocks per decile:")
    for d in range(1, amount_deciles+1):
        if d in decile_counts.index:
            print(f"      Decile {d:2d}: {decile_counts[d]:6.1f} stocks")

    # ========================================================================
    # STEP 6: CREATE HOLDING PERIOD CALENDAR
    # ========================================================================
    print(f"\n[6/8] Creating {K}-month holding period calendar...")

    # Define precise holding period boundaries
    # This ensures proper alignment with monthly returns
    signal['date'] = pd.to_datetime(signal['date'])
    signal['medate'] = signal['date'] + MonthEnd(0)      # Month-end date
    signal['hdate1'] = signal['medate'] + MonthBegin(1)  # First day of holding
    signal['hdate2'] = signal['medate'] + MonthEnd(K)    # Last day of holding
    
    # Prepare signal data for merging
    signal = signal[['permno', 'date', 'momr', 'hdate1', 'hdate2']].rename(
        columns={'date': 'form_date'}
    )

    # ========================================================================
    # STEP 7: MERGE WITH RETURN DATA
    # ========================================================================
    print(f"\n[7/8] Merging with returns data...")

    # Prepare returns data for holding period
    returns = data[['permno', 'date', 'ret', 'spread_open', 'spread_close', 'current_spread']].copy()
    returns['date'] = pd.to_datetime(returns['date'])
    
    # Focus on extreme deciles for long-short portfolio
    # This is standard in momentum literature
    is_extreme_decile = signal['momr'].isin([1, amount_deciles])
    signal = signal.loc[is_extreme_decile]

    # Efficient merge using chunking for large datasets
    port_chunks = []
    chunk_size = 100000
    
    # Sort for efficient merging
    signal = signal.sort_values(['permno', 'hdate1'])
    returns = returns.sort_values(['permno', 'date'])
    
    # Process in chunks to manage memory
    for i in range(0, len(returns), chunk_size):
        chunk = returns.iloc[i:i + chunk_size]
        
        # Merge signal with returns
        merged = signal.merge(chunk, on='permno', how='inner')
        
        # Keep only observations within holding period
        date_mask = (merged['hdate1'] <= merged['date']) & (merged['date'] <= merged['hdate2'])
        filtered_chunk = merged[date_mask]
        
        if len(filtered_chunk) > 0:
            port_chunks.append(filtered_chunk)
    
    # Combine all chunks
    if port_chunks:
        port = pd.concat(port_chunks, ignore_index=True)
    else:
        port = pd.DataFrame()

    print(f"   ✓ Created {len(port):,} stock-month holding observations")

    # ========================================================================
    # STEP 8: ADD TRANSACTION COSTS AND MARKET STATES
    # ========================================================================
    print(f"\n[8/8] Adding transaction costs and market states...")

    if len(port) == 0:
        return port, {}

    # Create date strings for commission mapping
    port['open_date'] = port['form_date'].dt.to_period('M').astype(str)
    port['current_date'] = port['date'].dt.to_period('M').astype(str)
    port['previous_date'] = (port['date'].dt.to_period('M') - 1).astype(str)

    # Map historical commission rates
    port['commission_open'] = port['open_date'].map(comm_map)
    port['commission_close'] = port['current_date'].map(comm_map)

    # Calculate market states for strategy logic
    market_state = compute_market_state(crsp_index_data, lags=market_state_lag, 
                                       column=market_state_column)
    market_state['merge_date'] = market_state['date'].dt.to_period('M').astype(str)
    state_map = market_state.set_index('merge_date')['market_state'].to_dict()
    
    # Add market states for different time periods
    port['hdate1_month'] = port['hdate1'].dt.to_period('M').astype(str)
    port['hdate2_month'] = port['hdate2'].dt.to_period('M').astype(str)
    port['form_market_state'] = port['hdate1_month'].map(state_map)      # State at formation
    port['current_market_state'] = port['current_date'].map(state_map)   # Current state
    port['prev_market_state'] = port['previous_date'].map(state_map)     # Previous month state

    # ========================================================================
    # CALCULATE STRATEGY RETURNS
    # ========================================================================
    # [The rest of the function continues with the strategy return calculations...]
    
    portfolio = port.copy()

    # Define position types and market states
    is_loser = portfolio['momr'] == 1
    is_winner = portfolio['momr'] == amount_deciles
    form_positive_state = portfolio['form_market_state'] == 1
    form_negative_state = portfolio['form_market_state'] == -1

    # Determine if market state has switched
    if adapt_state_switch:
        state_switched = (portfolio['form_market_state'] != portfolio['current_market_state'])
    else:
        state_switched = False  # No adaptation if adapt_state_switch is False

    # ------------------------------------------------------------------------
    # Strategy 0: Raw momentum returns (no transaction costs)
    # ------------------------------------------------------------------------
    # This is the baseline momentum strategy: long winners, short losers
    portfolio['RET_S0'] = np.select([
        is_winner,          
        is_loser
    ],[
        portfolio['ret'],   # Long winners
        -portfolio['ret']   # Short losers
    ])

    # ------------------------------------------------------------------------
    # Strategy 1: Market state reversal strategy (no transaction costs)
    # ------------------------------------------------------------------------
    # Implements conditional momentum/reversal based on market state
    # UP market: momentum (long winners, short losers)
    # DOWN market: reversal (long losers, short winners)
    if not transactional_framework or equal_comparision:
        # Create boolean series for state conditions
        state_switched_series = portfolio['form_market_state'] != portfolio['current_market_state'] if adapt_state_switch else pd.Series(False, index=portfolio.index)
        
        if close_positions:
            # Conservative approach: close positions on state switch
            portfolio['RET_S1'] = np.select([
                # Winners in positive state: long (unless state switched)
                is_winner & form_positive_state & ~state_switched_series,    
                # Winners in negative state: short
                is_winner & form_negative_state & ~state_switched_series,
                # Losers in positive state: short
                is_loser & form_positive_state & ~state_switched_series,
                # Losers in negative state: long
                is_loser & form_negative_state & ~state_switched_series,
                # Close all positions on state switch
                state_switched_series
            ],[
                portfolio['ret'],    # Long
                -portfolio['ret'],   # Short
                -portfolio['ret'],   # Short
                portfolio['ret'],    # Long
                0,  # Close position
            ])
        else:
            # Aggressive approach: reverse positions on state switch
            portfolio['RET_S1'] = np.select([
                # Standard positions (no switch)
                is_winner & form_positive_state & ~state_switched_series,    
                is_winner & form_negative_state & ~state_switched_series,
                is_winner & form_positive_state & state_switched_series,
                is_winner & form_negative_state & state_switched_series,   

                is_loser & form_positive_state & ~state_switched_series,
                is_loser & form_negative_state & ~state_switched_series,
                is_loser & form_positive_state & state_switched_series,
                is_loser & form_negative_state & state_switched_series,
            ],[
                # Standard positions
                portfolio['ret'],    # Long
                -portfolio['ret'],   # Short
                -portfolio['ret'],   # Reverse to short
                portfolio['ret'],    # Reverse to long

                -portfolio['ret'],   # Short
                portfolio['ret'],    # Long
                portfolio['ret'],    # Reverse to long
                -portfolio['ret'],   # Reverse to short
            ])

    # ------------------------------------------------------------------------
    # Handle state switching logic for transaction cost strategies
    # ------------------------------------------------------------------------
    if adapt_state_switch:
        # Track state switches and position closures
        portfolio['state_switched'] = (portfolio['form_market_state'] != portfolio['current_market_state'])
        
        # Sort to ensure chronological order
        portfolio = portfolio.sort_values(['permno', 'form_date', 'date'])
        
        # Create unique position identifier
        portfolio['position_id'] = portfolio['permno'].astype(str) + '_' + portfolio['form_date'].astype(str)
        
        # Detect first month of state switch for each position
        shifted = portfolio.groupby('position_id')['state_switched'].shift(1)
        shifted_nullable = shifted.astype("boolean")
        portfolio['prev_state_switched'] = shifted_nullable.fillna(False)
        portfolio['state_switch_month'] = portfolio['state_switched'] & ~portfolio['prev_state_switched']
        
        # Track positions closed by state switch
        if close_positions:
            # Positions are closed when state switches
            portfolio['switch_occurred_cumsum'] = portfolio.groupby('position_id')['state_switch_month'].cumsum()
            # All months after switch have closed_by_switch=True (return=0)
            portfolio['closed_by_switch'] = portfolio['switch_occurred_cumsum'] > 0
            # The switch month itself still gets return with closing costs
            portfolio.loc[portfolio['state_switch_month'], 'closed_by_switch'] = False
        else:
            # Positions are reversed, not closed
            portfolio['closed_by_switch'] = False
        
        # Clean up temporary columns
        portfolio = portfolio.drop(['position_id', 'prev_state_switched', 'switch_occurred_cumsum'], axis=1, errors='ignore')
    else:
        # No state switching when adapt_state_switch=False
        portfolio['state_switched'] = False
        portfolio['state_switch_month'] = False
        portfolio['closed_by_switch'] = False

    # ------------------------------------------------------------------------
    # Calculate transaction costs with proper scaling
    # ------------------------------------------------------------------------
    if transactional_framework:
        # Transaction costs scale with position value after return
        # This ensures realistic cost modeling
        closing_commission_adj = portfolio['commission_close'] + portfolio['spread_close']
        closing_cost_multiplier = 1 + portfolio['ret']  # Capital after return

        # Identify position lifecycle stages
        is_open = portfolio['current_date'] == portfolio['hdate1_month']
        is_working = (portfolio['current_date'] != portfolio['hdate1_month']) & \
                     (portfolio['current_date'] != portfolio['hdate2_month'])
        is_close = portfolio['current_date'] == portfolio['hdate2_month']

        # ------------------------------------------------------------------------
        # Strategy 2: Standard momentum with transaction costs
        # ------------------------------------------------------------------------
        # Transaction costs at opening reduce initial capital
        # Transaction costs at closing are paid on final capital
        loser_open_cost = -portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        loser_close_cost = -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        winner_open_cost = portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        winner_close_cost = portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        
        portfolio['RET_S2'] = np.select([
            is_loser & is_open,     # Open short position
            is_loser & is_close,    # Close short position
            is_loser & is_working,  # Hold short position
            is_winner & is_open,    # Open long position
            is_winner & is_close,   # Close long position
            is_winner & is_working  # Hold long position
        ], [
            loser_open_cost,        
            loser_close_cost,       
            -portfolio['ret'],      # No transaction costs while holding
            winner_open_cost,       
            winner_close_cost,       
            portfolio['ret']        # No transaction costs while holding
        ])

        # ------------------------------------------------------------------------
        # Strategy 3: Market state-aware strategy with transaction costs
        # ------------------------------------------------------------------------
        # This is the full conditional strategy with all features
        
        # === STANDARD TRANSACTION COSTS ===
        
        # Opening costs for different market states
        pos_state_winner_open_cost = portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        pos_state_loser_open_cost = -portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        neg_state_winner_open_cost = -portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        neg_state_loser_open_cost = portfolio['ret'] - (portfolio['commission_open'] + portfolio['spread_open']) * (1 + portfolio['ret'])
        
        # Working period returns (no transaction costs)
        pos_state_winner_working_cost = portfolio['ret']
        pos_state_loser_working_cost = -portfolio['ret']
        neg_state_winner_working_cost = -portfolio['ret']
        neg_state_loser_working_cost = portfolio['ret']
        
        # Closing costs
        pos_state_winner_close_cost = portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        pos_state_loser_close_cost = -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        neg_state_winner_close_cost = -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        neg_state_loser_close_cost = portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        
        # === SPECIAL CASES FOR STATE SWITCHES ===
        
        # State switch closing costs (for mid-holding period switches)
        switch_close_winner_pos = portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        switch_close_loser_pos = -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        switch_close_winner_neg = -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        switch_close_loser_neg = portfolio['ret'] - closing_commission_adj * closing_cost_multiplier
        
        # State switch reversal costs (close existing + open opposite position)
        switch_reverse_winner_pos_to_neg = (
            portfolio['ret'] - closing_commission_adj * closing_cost_multiplier  # Close long
            - (portfolio['commission_open'] + portfolio['spread_open'])          # Open short
        )
        switch_reverse_loser_pos_to_neg = (
            -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier  # Close short
            - (portfolio['commission_open'] + portfolio['spread_open'])           # Open long
        )
        switch_reverse_winner_neg_to_pos = (
            -portfolio['ret'] - closing_commission_adj * closing_cost_multiplier  # Close short
            - (portfolio['commission_open'] + portfolio['spread_open'])           # Open long
        )
        switch_reverse_loser_neg_to_pos = (
            portfolio['ret'] - closing_commission_adj * closing_cost_multiplier   # Close long
            - (portfolio['commission_open'] + portfolio['spread_open'])           # Open short
        )
        
        # === EDGE CASE: State switch in opening month ===
        # These require special handling as they include initial opening costs
        
        # If close_positions=True: pay opening + closing costs
        switch_open_close_winner_pos = (
            portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
        )
        switch_open_close_loser_pos = (
            -portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
        )
        switch_open_close_winner_neg = (
            -portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
        )
        switch_open_close_loser_neg = (
            portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
        )
        
        # If close_positions=False: pay opening + closing + reverse opening costs
        switch_open_reverse_winner_pos_to_neg = (
            portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Reverse opening
        )
        switch_open_reverse_loser_pos_to_neg = (
            -portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Reverse opening
        )
        switch_open_reverse_winner_neg_to_pos = (
            -portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Reverse opening
        )
        switch_open_reverse_loser_neg_to_pos = (
            portfolio['ret'] 
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Initial opening
            - closing_commission_adj * closing_cost_multiplier           # Closing
            - (portfolio['commission_open'] + portfolio['spread_open'])  # Reverse opening
        )
        
        # Re-define boolean conditions for clarity
        is_loser = portfolio['momr'] == 1
        is_winner = portfolio['momr'] == amount_deciles
        form_positive_state = portfolio['form_market_state'] == 1
        form_negative_state = portfolio['form_market_state'] == -1
        is_open = portfolio['current_date'] == portfolio['hdate1_month']
        is_working = (portfolio['current_date'] != portfolio['hdate1_month']) & \
                     (portfolio['current_date'] != portfolio['hdate2_month'])
        is_close = portfolio['current_date'] == portfolio['hdate2_month']

        # State switch conditions for working period
        state_switch_conditions = {
            'winner_pos_switch': is_winner & form_positive_state & portfolio['state_switch_month'] & is_working,
            'winner_neg_switch': is_winner & form_negative_state & portfolio['state_switch_month'] & is_working,
            'loser_pos_switch': is_loser & form_positive_state & portfolio['state_switch_month'] & is_working,
            'loser_neg_switch': is_loser & form_negative_state & portfolio['state_switch_month'] & is_working,
        }

        # Conditions for continuing after state switch (reversed positions)
        state_switch_continue_conditions = {
            'winner_pos_continue': is_winner & form_positive_state & portfolio['state_switched'] & ~portfolio['state_switch_month'] & is_working,
            'winner_neg_continue': is_winner & form_negative_state & portfolio['state_switched'] & ~portfolio['state_switch_month'] & is_working,
            'loser_pos_continue': is_loser & form_positive_state & portfolio['state_switched'] & ~portfolio['state_switch_month'] & is_working,
            'loser_neg_continue': is_loser & form_negative_state & portfolio['state_switched'] & ~portfolio['state_switch_month'] & is_working,
        }

        # Build conditions and choices based on close_positions setting
        if close_positions:
            # Conservative approach: close positions on state switch
            conditions = [
                # === STANDARD CASES (NO STATE SWITCH) ===
                
                # Opening positions
                is_winner & form_positive_state & is_open & ~portfolio['state_switched'],
                is_winner & form_negative_state & is_open & ~portfolio['state_switched'],
                is_loser & form_positive_state & is_open & ~portfolio['state_switched'],
                is_loser & form_negative_state & is_open & ~portfolio['state_switched'],
                
                # === EDGE CASE: State switch in opening month ===
                
                # Close only (close_positions=True)
                is_winner & form_positive_state & is_open & portfolio['state_switched'],
                is_winner & form_negative_state & is_open & portfolio['state_switched'],
                is_loser & form_positive_state & is_open & portfolio['state_switched'],
                is_loser & form_negative_state & is_open & portfolio['state_switched'],
                
                # === HOLDING PERIOD (NO STATE SWITCH) ===
                
                is_winner & form_positive_state & ~portfolio['state_switched'] & is_working,
                is_winner & form_negative_state & ~portfolio['state_switched'] & is_working,
                is_loser & form_positive_state & ~portfolio['state_switched'] & is_working,
                is_loser & form_negative_state & ~portfolio['state_switched'] & is_working,
                
                # === NORMAL CLOSING (NO STATE SWITCH) ===
                
                is_winner & form_positive_state & ~portfolio['state_switched'] & is_close,
                is_winner & form_negative_state & ~portfolio['state_switched'] & is_close,
                is_loser & form_positive_state & ~portfolio['state_switched'] & is_close,
                is_loser & form_negative_state & ~portfolio['state_switched'] & is_close,
                
                # === EDGE CASE: State switch in closing month ===
                
                is_winner & form_positive_state & portfolio['state_switched'] & is_close,
                is_winner & form_negative_state & portfolio['state_switched'] & is_close,
                is_loser & form_positive_state & portfolio['state_switched'] & is_close,
                is_loser & form_negative_state & portfolio['state_switched'] & is_close,
                
                # === STATE SWITCH DURING HOLDING PERIOD ===
                
                # First month of switch: close positions
                state_switch_conditions['winner_pos_switch'],
                state_switch_conditions['winner_neg_switch'],
                state_switch_conditions['loser_pos_switch'],
                state_switch_conditions['loser_neg_switch'],
                
                # Subsequent months after closing by switch
                portfolio['closed_by_switch'] & ~portfolio['state_switch_month'],
            ]
            
            choices = [
                # Standard opening positions
                pos_state_winner_open_cost,
                neg_state_winner_open_cost,
                pos_state_loser_open_cost,
                neg_state_loser_open_cost,
                
                # Edge case: State switch in opening month (close only)
                switch_open_close_winner_pos,
                switch_open_close_winner_neg,
                switch_open_close_loser_pos,
                switch_open_close_loser_neg,
                
                # Normal holding period
                pos_state_winner_working_cost,
                neg_state_winner_working_cost,
                pos_state_loser_working_cost,
                neg_state_loser_working_cost,
                
                # Normal closing
                pos_state_winner_close_cost,
                neg_state_winner_close_cost,
                pos_state_loser_close_cost,
                neg_state_loser_close_cost,
                
                # Edge case: State switch in closing month
                pos_state_winner_close_cost,
                neg_state_winner_close_cost,
                pos_state_loser_close_cost,
                neg_state_loser_close_cost,
                
                # State switch during holding - close positions
                switch_close_winner_pos,
                switch_close_winner_neg,
                switch_close_loser_pos,
                switch_close_loser_neg,
                
                # Closed by switch - subsequent months
                0,
            ]
        else:  # close_positions = False
            # Aggressive approach: reverse positions on state switch
            conditions = [
                # === STANDARD CASES (NO STATE SWITCH) ===
                
                # Opening positions
                is_winner & form_positive_state & is_open & ~portfolio['state_switched'],
                is_winner & form_negative_state & is_open & ~portfolio['state_switched'],
                is_loser & form_positive_state & is_open & ~portfolio['state_switched'],
                is_loser & form_negative_state & is_open & ~portfolio['state_switched'],
                
                # === EDGE CASE: State switch in opening month ===
                
                # Close and reverse (close_positions=False)
                is_winner & form_positive_state & is_open & portfolio['state_switched'],
                is_winner & form_negative_state & is_open & portfolio['state_switched'],
                is_loser & form_positive_state & is_open & portfolio['state_switched'],
                is_loser & form_negative_state & is_open & portfolio['state_switched'],
                
                # === HOLDING PERIOD (NO STATE SWITCH) ===
                
                is_winner & form_positive_state & ~portfolio['state_switched'] & is_working,
                is_winner & form_negative_state & ~portfolio['state_switched'] & is_working,
                is_loser & form_positive_state & ~portfolio['state_switched'] & is_working,
                is_loser & form_negative_state & ~portfolio['state_switched'] & is_working,
                
                # === NORMAL CLOSING (NO STATE SWITCH) ===
                
                is_winner & form_positive_state & ~portfolio['state_switched'] & is_close,
                is_winner & form_negative_state & ~portfolio['state_switched'] & is_close,
                is_loser & form_positive_state & ~portfolio['state_switched'] & is_close,
                is_loser & form_negative_state & ~portfolio['state_switched'] & is_close,
                
                # === EDGE CASE: State switch in closing month ===
                
                is_winner & form_positive_state & portfolio['state_switched'] & is_close,
                is_winner & form_negative_state & portfolio['state_switched'] & is_close,
                is_loser & form_positive_state & portfolio['state_switched'] & is_close,
                is_loser & form_negative_state & portfolio['state_switched'] & is_close,
                
                # === STATE SWITCH DURING HOLDING PERIOD ===
                
                # First month of switch: reverse positions
                state_switch_conditions['winner_pos_switch'],
                state_switch_conditions['winner_neg_switch'],
                state_switch_conditions['loser_pos_switch'],
                state_switch_conditions['loser_neg_switch'],
                
                # Continuing with reversed positions
                state_switch_continue_conditions['winner_pos_continue'],
                state_switch_continue_conditions['winner_neg_continue'],
                state_switch_continue_conditions['loser_pos_continue'],
                state_switch_continue_conditions['loser_neg_continue'],
            ]
            
            choices = [
                # Standard opening positions
                pos_state_winner_open_cost,
                neg_state_winner_open_cost,
                pos_state_loser_open_cost,
                neg_state_loser_open_cost,
                
                # Edge case: State switch in opening month (close and reverse)
                switch_open_reverse_winner_pos_to_neg,
                switch_open_reverse_winner_neg_to_pos,
                switch_open_reverse_loser_pos_to_neg,
                switch_open_reverse_loser_neg_to_pos,
                
                # Normal holding period
                pos_state_winner_working_cost,
                neg_state_winner_working_cost,
                pos_state_loser_working_cost,
                neg_state_loser_working_cost,
                
                # Normal closing
                pos_state_winner_close_cost,
                neg_state_winner_close_cost,
                pos_state_loser_close_cost,
                neg_state_loser_close_cost,
                
                # Edge case: State switch in closing month
                pos_state_winner_close_cost,
                neg_state_winner_close_cost,
                pos_state_loser_close_cost,
                neg_state_loser_close_cost,
                
                # State switch during holding - reverse positions (first month)
                switch_reverse_winner_pos_to_neg,
                switch_reverse_winner_neg_to_pos,
                switch_reverse_loser_pos_to_neg,
                switch_reverse_loser_neg_to_pos,
                
                # Continuing reversed positions
                neg_state_winner_working_cost,  # Was pos winner, now acts as neg winner
                pos_state_winner_working_cost,  # Was neg winner, now acts as pos winner
                neg_state_loser_working_cost,   # Was pos loser, now acts as neg loser
                pos_state_loser_working_cost,   # Was neg loser, now acts as pos loser
            ]

        # Apply the strategy
        portfolio['RET_S3'] = np.select(conditions, choices, default=0)

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    print(f"\n✓ Strategy execution complete")
    print(f"✓ Portfolio contains {len(portfolio):,} stock-month observations")
    print(f"✓ Market cap constraint: bottom {MKT_CAP_CUTOFF*100:.0f}% of stocks by size")
    print("=" * 80)

    return portfolio, state_map


def apply_rebalancing_costs_balanced(portfolio, adapt_state_switch, close_positions, 
                                   return_column='RET_S3'):
    """
    Apply monthly rebalancing costs to maintain equal-weighted long-short portfolio.
    
    This function calculates the costs associated with rebalancing positions to maintain
    target weights (50% long, 50% short, equally weighted within each side). The function
    implements realistic constraints on trade sizes and costs to prevent excessive trading.
    
    Rebalancing Logic:
    1. Each month, calculate current position values and weights
    2. Determine target weights (equal weight within long/short sides)
    3. Calculate required trades to reach targets
    4. Apply realistic constraints on trade sizes and costs
    5. Deduct costs from returns
    
    Parameters
    ----------
    portfolio : pd.DataFrame
        Portfolio data from run_momentum_strategy containing positions and returns
        
    adapt_state_switch : bool
        Whether positions adapt to market state switches (from parent strategy)
        
    close_positions : bool
        Whether positions are closed on state switch (True) or reversed (False)
        
    return_column : str, default='RET_S3'
        Column name containing the strategy returns to adjust
        
    Returns
    -------
    portfolio : pd.DataFrame
        Original portfolio DataFrame with additional columns:
        - position_value: Value of position after return
        - current_weight: Current portfolio weight of position
        - target_weight: Target portfolio weight (for rebalancing)
        - rebalancing_trade_fraction: Fraction of position to trade (capped at 40%)
        - rebalancing_cost: Cost of rebalancing (capped at 30 bps)
        - {return_column}_rebal: Returns after rebalancing costs
        
    Notes
    -----
    Constraints:
    - Maximum trade size: 40% of position value (prevents extreme trades)
    - Maximum rebalancing cost: 30 basis points (prevents excessive costs)
    
    These constraints reflect realistic market conditions where large trades
    may face liquidity constraints and price impact.
    """
    
    # ========================================================================
    # CONSTANTS AND INITIALIZATION
    # ========================================================================
    
    MAX_TRADE_FRACTION = 0.40  # Maximum 40% of position can be traded
    MAX_REBAL_COST = 0.003     # Maximum 30 basis points rebalancing cost
    
    # Identify position lifecycle stages
    is_open = portfolio['current_date'] == portfolio['hdate1_month']
    is_close = portfolio['current_date'] == portfolio['hdate2_month']
    is_working = ~is_open & ~is_close
    
    # Determine active positions (not closed by state switch)
    if adapt_state_switch and close_positions:
        # Positions may be closed early due to state switch
        active_mask = (is_open | is_working) & ~portfolio['closed_by_switch']
    else:
        # All open and working positions are active
        active_mask = is_open | is_working
    
    # ========================================================================
    # DETERMINE POSITION DIRECTIONS
    # ========================================================================
    
    # Identify winners and losers
    is_winner = portfolio['momr'] == portfolio['momr'].max()
    is_loser = portfolio['momr'] == 1
    
    # Identify market states
    form_positive_state = portfolio['form_market_state'] == 1
    form_negative_state = portfolio['form_market_state'] == -1
    
    # Track reversed positions (when adapt_state_switch=True and close_positions=False)
    if adapt_state_switch and not close_positions:
        # Positions are reversed after state switch (excluding the switch month itself)
        portfolio['position_reversed'] = (
            portfolio['state_switched'] & 
            ~portfolio['state_switch_month']
        )
    else:
        portfolio['position_reversed'] = False
    
    # Calculate position direction (1 for long, -1 for short)
    # This accounts for both original and reversed positions
    portfolio['position_direction'] = np.select([
        # Original positions (not reversed)
        (is_winner & form_positive_state & ~portfolio['position_reversed']),  # Long
        (is_loser & form_positive_state & ~portfolio['position_reversed']),   # Short
        (is_winner & form_negative_state & ~portfolio['position_reversed']),  # Short
        (is_loser & form_negative_state & ~portfolio['position_reversed']),   # Long
        
        # Reversed positions
        (is_winner & form_positive_state & portfolio['position_reversed']),   # Was long, now short
        (is_loser & form_positive_state & portfolio['position_reversed']),    # Was short, now long
        (is_winner & form_negative_state & portfolio['position_reversed']),   # Was short, now long
        (is_loser & form_negative_state & portfolio['position_reversed']),    # Was long, now short
    ], [
        1, -1, -1, 1,    # Original directions
        -1, 1, 1, -1     # Reversed directions
    ], default=0)
    
    # ========================================================================
    # INITIALIZE REBALANCING COLUMNS
    # ========================================================================
    
    portfolio['position_value'] = 0.0
    portfolio['current_weight'] = 0.0
    portfolio['target_weight'] = 0.0
    portfolio['rebalancing_trade_fraction'] = 0.0
    portfolio['rebalancing_cost'] = 0.0
    
    # ========================================================================
    # CALCULATE REBALANCING BY DATE
    # ========================================================================
    
    # Process each date separately to calculate portfolio weights
    for date in sorted(portfolio['date'].unique()):
        date_mask = portfolio['date'] == date
        
        # Calculate position values
        # Active positions: value = 1 + return (assuming $1 initial investment)
        # Inactive positions: value = 1 (no exposure)
        portfolio.loc[date_mask & active_mask, 'position_value'] = (
            1 + portfolio.loc[date_mask & active_mask, return_column]
        )
        portfolio.loc[date_mask & ~active_mask, 'position_value'] = 1
        
        # Count positions by direction
        long_mask = date_mask & (portfolio['position_direction'] == 1)
        short_mask = date_mask & (portfolio['position_direction'] == -1)
        
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        
        if n_long + n_short == 0:
            continue  # No positions to rebalance
        
        # Calculate current portfolio weights
        total_value = portfolio.loc[date_mask, 'position_value'].sum()
        
        if total_value > 0:
            portfolio.loc[date_mask, 'current_weight'] = (
                portfolio.loc[date_mask, 'position_value'] / total_value
            )
        else:
            continue  # Skip if total value is non-positive
        
        # Set target weights
        # Goal: 50% in long positions, 50% in short positions
        # Within each side, positions are equally weighted
        if n_long > 0:
            portfolio.loc[long_mask, 'target_weight'] = 0.5 / n_long
        if n_short > 0:
            portfolio.loc[short_mask, 'target_weight'] = 0.5 / n_short
        
        # ========================================================================
        # CALCULATE REBALANCING TRADES AND COSTS
        # ========================================================================
        
        # Only rebalance active positions
        active_date_mask = date_mask & active_mask
        
        if active_date_mask.sum() > 0:
            # Select positions that need rebalancing (non-zero current weight)
            rebal_mask = active_date_mask & (portfolio['current_weight'] > 0)
            
            if rebal_mask.sum() > 0:
                # Calculate required trade fraction to reach target weight
                # trade_fraction = (target - current) / current
                # Positive: buy more, Negative: sell some
                raw_trade = (
                    (portfolio.loc[rebal_mask, 'target_weight'] - 
                     portfolio.loc[rebal_mask, 'current_weight']) / 
                    portfolio.loc[rebal_mask, 'current_weight']
                )
                
                # Apply maximum trade size constraint (40% of position)
                # This prevents unrealistic large trades
                portfolio.loc[rebal_mask, 'rebalancing_trade_fraction'] = np.clip(
                    raw_trade, -MAX_TRADE_FRACTION, MAX_TRADE_FRACTION
                )
                
                # Calculate rebalancing cost
                # Cost = |trade_fraction| * (commission + half_spread)
                raw_cost = (
                    np.abs(portfolio.loc[rebal_mask, 'rebalancing_trade_fraction']) * 
                    (portfolio.loc[rebal_mask, 'commission_close'] + 
                     portfolio.loc[rebal_mask, 'current_spread'])
                )
                
                # Apply maximum cost constraint (30 basis points)
                # This prevents excessive costs from dominating returns
                portfolio.loc[rebal_mask, 'rebalancing_cost'] = np.minimum(
                    raw_cost, MAX_REBAL_COST
                )
    
    # ========================================================================
    # APPLY REBALANCING COSTS TO RETURNS
    # ========================================================================
    
    # Create new column for returns after rebalancing
    portfolio[f'{return_column}_rebal'] = portfolio[return_column].copy()
    
    # Deduct rebalancing costs from active positions only
    # Inactive positions keep their original returns (likely 0)
    portfolio.loc[active_mask, f'{return_column}_rebal'] = (
        portfolio.loc[active_mask, return_column] - 
        portfolio.loc[active_mask, 'rebalancing_cost']
    )
    
    return portfolio


def analyze_momentum_portfolio(port_data, market_state_map, 
                             start_date='1940-01-01', end_date='2025-04-30',
                             return_column='RET_S1', K=1, market_state_lag=24,
                             amount_deciles=10):
    """
    Analyze momentum portfolio performance with statistical inference.
    
    This function computes portfolio returns following the methodology of
    Jegadeesh and Titman (1993) with overlapping portfolio adjustments:
    
    1. Equal-weight averaging across cohorts (1/K weighting)
    2. Newey-West standard errors accounting for both holding period and market state lag
    3. Risk-adjusted performance metrics (Sharpe ratio, maximum drawdown)
    
    The overlapping portfolio approach means that at any given time, K different
    cohorts are active (formed in the past K months). We equal-weight across
    these cohorts to reduce noise and improve statistical power.
    
    IMPORTANT: HAC lag structure now uses K + market_state_lag - 1 to properly
    account for autocorrelation from both overlapping portfolios and market state
    persistence.
    
    Parameters
    ----------
    port_data : pd.DataFrame
        Portfolio data from run_momentum_strategy() containing columns:
        - date: Trading date
        - momr: Momentum decile rank (1=losers, 10=winners)
        - form_date: Portfolio formation date
        - return_column: Strategy returns
        
    market_state_map : dict
        Dictionary mapping month strings (YYYY-MM) to market states (1, 0, -1)
        Used for conditional analysis and visualization
        
    start_date : str, default='1940-01-01'
        Analysis start date in YYYY-MM-DD format
        Allows for subsample analysis
        
    end_date : str, default='2025-04-30'
        Analysis end date in YYYY-MM-DD format
        
    return_column : str, default='RET_S1'
        Column name containing strategy returns to analyze
        Options: 
        - 'RET_S0': Raw momentum without costs
        - 'RET_S1': Market state-aware without costs
        - 'RET_S2': Standard momentum with costs
        - 'RET_S3': Full conditional strategy with costs
        
    K : int, default=1
        Holding period length in months
        Critical for determining HAC lag structure
        
    market_state_lag : int, default=24
        Market state lookback period in months
        Used for proper HAC lag calculation: HAC_lags = K + market_state_lag - 1
        
    amount_deciles : int, default=10
        Number of momentum deciles used in portfolio formation
        Determines winner (highest) and loser (lowest) portfolios
        
    Returns
    -------
    dict
        Comprehensive analysis results containing:
        - decile_ret : pd.DataFrame
            Time series of portfolio returns with columns:
            - winners: Top decile returns
            - losers: Bottom decile returns
            - long_short: Long-short portfolio returns
            - cum_val_long_short: Cumulative value of $1 invested
            - Various log return measures
            - market_state: Current market regime
            
        - t_stat : float
            HAC-adjusted t-statistic for mean return
            Uses Newey-West with K + market_state_lag - 1 lags
            
        - p_value : float
            Two-sided p-value for significance test
            H0: mean return = 0
            
        - avg_monthly_return : float
            Average arithmetic monthly return
            
        - sharpe_ratio : float
            Annualized Sharpe ratio (assuming zero risk-free rate)
            SR = mean_return / volatility * sqrt(12)
            
        - max_drawdown : float
            Maximum peak-to-trough decline (negative value)
            MDD = min((V_t - max(V_s, s≤t)) / max(V_s, s≤t))
            
        - monthly_vol : float
            Monthly return standard deviation
        
    Notes
    -----
    Statistical Methodology:
    - The function adjusts for overlapping portfolios by averaging across
      all active cohorts at each point in time (1/K weighting)
    - T-statistics use Newey-West standard errors with K + market_state_lag - 1 lags
      to account for serial correlation from both overlapping holding periods
      and market state persistence
    - This extends the original Jegadeesh and Titman (1993) approach to
      accommodate market state-dependent strategies
    
    Performance Metrics:
    - Sharpe ratio assumes zero risk-free rate (excess returns = raw returns)
    - Maximum drawdown calculated on cumulative return series
    - All metrics computed on the long-short portfolio (winners - losers)
    
    References
    ----------
    Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers:
    Implications for stock market efficiency. The Journal of Finance, 48(1), 65-91.
    """
    
    # ========================================================================
    # INITIALIZATION AND DATA FILTERING
    # ========================================================================
    
    print(f"\n" + "-" * 80)
    print(f"ANALYZING PORTFOLIO: {return_column}")
    print("-" * 80)
    
    # Filter data to specified analysis period
    # This allows for subsample analysis and out-of-sample testing
    print(f"Filtering data from {start_date} to {end_date}...")
    filtered_data = port_data[
        (port_data['date'] >= start_date) & 
        (port_data['date'] <= end_date)
    ].copy()
    
    print(f"   ✓ Selected {len(filtered_data):,} observations")
    
    # ========================================================================
    # CALCULATE EQUAL-WEIGHT PORTFOLIO RETURNS
    # ========================================================================
    print("\nCalculating equal-weight portfolio returns...")
    
    # ------------------------------------------------------------------------
    # Step 1: Average within each cohort-date-decile combination
    # ------------------------------------------------------------------------
    # This handles multiple stocks in the same decile by equal-weighting
    # within each momentum group at each point in time
    cohort_ret = (filtered_data
                  .groupby(['date', 'momr', 'form_date'])[return_column]
                  .mean()  # Equal-weight within decile
                  .reset_index())
    
    # ------------------------------------------------------------------------
    # Step 2: Average across overlapping cohorts (1/K adjustment)
    # ------------------------------------------------------------------------
    # At any given date, K different cohorts may be active
    # (formed in the past K months). We equal-weight across these cohorts
    # This is the standard Jegadeesh-Titman overlapping portfolio adjustment
    cohort_ret = (cohort_ret
                  .groupby(['date', 'momr'])[return_column]
                  .mean()  # Equal-weight across cohorts
                  .reset_index())
    
    # ------------------------------------------------------------------------
    # Step 3: Reshape to wide format for portfolio construction
    # ------------------------------------------------------------------------
    # Pivot data to have one column per momentum decile
    decile_ret = cohort_ret.pivot(index='date', columns='momr', values=return_column)
    decile_ret.columns = [f'port{c}' for c in decile_ret.columns]
    
    # ------------------------------------------------------------------------
    # Step 4: Construct momentum portfolios
    # ------------------------------------------------------------------------
    # Define winner-minus-loser (WML) portfolio
    # Note: In our convention, losers have negative returns (short position)
    # so long_short = winners + losers (not winners - losers)
    if 'port1' in decile_ret.columns and f'port{amount_deciles}' in decile_ret.columns:
        # Rename for clarity
        decile_ret = decile_ret.rename(columns={
            'port1': 'losers',                    # Lowest momentum decile
            f'port{amount_deciles}': 'winners'    # Highest momentum decile
        })
        
        # Create long-short portfolio
        # Note: losers already have negative sign from strategy construction
        decile_ret['long_short'] = decile_ret['winners'] + decile_ret['losers']
        print("   ✓ Created portfolios")
    else:
        # Handle case where extreme deciles might be missing in some periods
        print(f"   ⚠ Warning: Missing decile 1 or {amount_deciles} in some periods")
        decile_ret['long_short'] = np.nan
    
    # ========================================================================
    # CALCULATE CUMULATIVE PERFORMANCE METRICS
    # ========================================================================
    print("\nCalculating performance metrics...")
    
    # ------------------------------------------------------------------------
    # Cumulative returns in different representations
    # ------------------------------------------------------------------------
    # Standard cumulative value (multiplicative)
    # Shows value of $1 invested at inception
    decile_ret['cum_val_long_short'] = (1 + decile_ret['long_short']).cumprod()
    
    # Log returns for additive properties and statistical analysis
    # Log returns are more suitable for statistical tests due to better distributional properties
    decile_ret['log_long_short'] = np.log1p(decile_ret['long_short'])
    decile_ret['cum_log_long_short'] = decile_ret['log_long_short'].cumsum()
    
    # Log base 2 returns (for information-theoretic interpretations)
    # Useful for understanding doubling times
    decile_ret['log2_long_short'] = np.log2(1 + decile_ret['long_short'])
    decile_ret['cum_log2_long_short'] = decile_ret['log2_long_short'].cumsum()
    
    # ------------------------------------------------------------------------
    # Map market states for conditional analysis
    # ------------------------------------------------------------------------
    # Convert index to datetime and create merge key
    decile_ret['merge_date'] = pd.to_datetime(decile_ret.index)
    decile_ret['merge_date'] = decile_ret['merge_date'].dt.to_period('M').astype(str)
    
    # Map market states from provided dictionary
    decile_ret['market_state'] = decile_ret['merge_date'].map(market_state_map)
    
    # ========================================================================
    # STATISTICAL TESTING WITH HAC STANDARD ERRORS
    # ========================================================================
    print("\nPerforming statistical tests...")
    
    # Remove any NaN values for regression analysis
    valid_returns = decile_ret['long_short'].dropna()
    
    if len(valid_returns) > 0:
        # ------------------------------------------------------------------------
        # Calculate proper HAC lag structure
        # ------------------------------------------------------------------------
        # NEW: Account for both holding period and market state lag
        # This captures autocorrelation from overlapping portfolios AND
        # persistence in market state classification
        hac_lags = K + market_state_lag - 1
        
        print(f"   ✓ HAC lag calculation:")
        print(f"      - Holding period (K): {K} months")
        print(f"      - Market state lag: {market_state_lag} months")
        print(f"      - Total HAC lags: {hac_lags} months")
        
        # ------------------------------------------------------------------------
        # OLS regression with Newey-West HAC standard errors
        # ------------------------------------------------------------------------
        # Model: R_t = α + ε_t
        # Where α is the average return we want to test
        # H0: α = 0 (no abnormal returns)
        # H1: α ≠ 0 (significant momentum profits)
        
        # Use proper lag structure for market state-dependent strategies
        ols = sm.OLS(valid_returns,                      # Dependent variable
                     np.ones_like(valid_returns)          # Constant term only
                     ).fit(
                         cov_type='HAC',                  # Heteroskedasticity and Autocorrelation Consistent
                         cov_kwds={'maxlags': hac_lags}   # Extended lag structure
                     )
        
        # Extract test statistics
        t_stat = ols.tvalues.iloc[0]      # t-statistic for constant (mean return)
        p_value = ols.pvalues.iloc[0]     # Two-sided p-value
        avg_monthly_return = valid_returns.mean()  # Simple average return
        
        print(f"   ✓ Estimated with {len(valid_returns)} monthly observations")
        print(f"   ✓ Using {hac_lags} lags for HAC adjustment")
    else:
        # Insufficient data for statistical inference
        t_stat = np.nan
        p_value = np.nan
        avg_monthly_return = np.nan
        print("   ⚠ Insufficient data for statistical tests")
    
    # ========================================================================
    # RISK-ADJUSTED PERFORMANCE METRICS
    # ========================================================================
    print("\nCalculating risk metrics...")
    
    # ------------------------------------------------------------------------
    # Monthly volatility (standard deviation of returns)
    # ------------------------------------------------------------------------
    monthly_vol = valid_returns.std() if len(valid_returns) > 0 else np.nan
    
    # ------------------------------------------------------------------------
    # Sharpe ratio (annualized)
    # ------------------------------------------------------------------------
    # SR = (E[R] - Rf) / σ * √12
    # Assuming Rf = 0 (excess returns interpretation)
    # This is standard for long-short portfolios which are self-financing
    if not np.isnan(monthly_vol) and monthly_vol > 0:
        sharpe_ratio = (avg_monthly_return / monthly_vol) * np.sqrt(12)
    else:
        sharpe_ratio = np.nan
    
    # ------------------------------------------------------------------------
    # Maximum drawdown calculation
    # ------------------------------------------------------------------------
    # MDD = min((V_t - max(V_s, s≤t)) / max(V_s, s≤t))
    # Measures worst peak-to-trough decline
    # Critical for understanding tail risk
    if len(valid_returns) > 0:
        # Calculate cumulative value series
        cum_returns = (1 + valid_returns).cumprod()
        
        # Running maximum (peak values)
        running_max = cum_returns.expanding().max()
        
        # Drawdown series (percentage decline from peak)
        drawdown = (cum_returns - running_max) / running_max
        
        # Maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
    else:
        max_drawdown = np.nan
    
    print(f"   ✓ Computed Sharpe ratio and maximum drawdown")
    
    # ========================================================================
    # RETURN RESULTS DICTIONARY
    # ========================================================================
    return {
        'decile_ret': decile_ret,
        't_stat': t_stat,
        'p_value': p_value,
        'avg_monthly_return': avg_monthly_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'monthly_vol': monthly_vol
    }

def compute_bonferroni_corrected_statistics(return_series_dict, formation_periods, holding_periods, 
                                           market_state_lags, significance_level=0.01):
    """
    Compute Bonferroni-corrected t-statistics for multiple momentum strategy combinations.
    
    This function implements the eigenvalue-based Bonferroni correction methodology proposed
    by Harvey, Liu, and Zhu (2016) to address multiple testing concerns in strategy evaluation.
    The correction accounts for correlations between strategies by using principal component
    analysis to identify the effective number of independent tests.
    
    Methodology:
    1. Construct correlation matrix of strategy returns across all parameter combinations
    2. Perform eigenvalue decomposition to identify principal components
    3. Determine effective number of tests (k) such that first k eigenvalues explain ≥99% variance
    4. Apply Bonferroni correction: critical value = Φ^(-1)(1 - α/(2k))
    5. Compute HAC-adjusted t-statistics for each strategy combination
    
    This approach addresses the "multiple comparisons problem" common in quantitative finance
    where testing many strategies inflates Type I error rates, leading to false discoveries.
    
    Parameters
    ----------
    return_series_dict : dict
        Dictionary containing time series returns for each parameter combination.
        Structure: {(market_state_lag, J, K): pd.Series of returns}
        where J = formation period, K = holding period
        
    formation_periods : list
        List of formation period values (J) used in grid search
        
    holding_periods : list
        List of holding period values (K) used in grid search
        
    market_state_lags : list
        List of market state lag values used for regime identification
        
    significance_level : float, default=0.05
        Nominal significance level before correction (typically 0.05 for 5% level)
        
    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        - 'correlation_matrix': pd.DataFrame of strategy return correlations
        - 'eigenvalues': np.array of correlation matrix eigenvalues (descending order)
        - 'variance_explained': np.array of cumulative variance explained by eigenvalues
        - 'effective_tests': int, number of effective independent tests (k)
        - 'critical_value': float, Bonferroni-corrected critical t-value
        - 'raw_critical': float, uncorrected critical t-value (±1.96 for α=0.05)
        - 'individual_results': dict, detailed results for each parameter combination
        - 'significant_strategies': list, strategies significant after correction
        - 'rejection_summary': dict, summary of rejections before/after correction
        
    Notes
    -----
    Statistical Framework:
    The correction addresses the fundamental problem that when testing m strategies
    simultaneously, the probability of at least one false positive is:
    P(Type I Error) = 1 - (1-α)^m ≈ m×α for small α
    
    For 48 strategies with α=0.05, this gives ~92% chance of false discovery without correction.
    
    The eigenvalue approach improves upon naive Bonferroni (critical = α/m) by recognizing
    that correlated strategies provide less independent information. If strategies are
    perfectly correlated, we effectively have only 1 test, not m tests.
    
    Critical Value Calculation:
    - Naive Bonferroni: t_crit = Φ^(-1)(1 - α/2m)
    - Eigenvalue-adjusted: t_crit = Φ^(-1)(1 - α/2k), where k ≤ m
    
    HAC Standard Errors:
    Each individual t-statistic uses Newey-West standard errors with lag structure:
    L = K + market_state_lag - 1
    This accounts for autocorrelation from overlapping portfolios and market state persistence.
    
    References
    ----------
    Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns.
    The Review of Financial Studies, 29(1), 5-68.
    """
    
    print("\n" + "=" * 80)
    print("BONFERRONI CORRECTION ANALYSIS FOR MULTIPLE TESTING")
    print("Eigenvalue-Based Approach Following Harvey, Liu, and Zhu (2016)")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: DATA PREPARATION AND VALIDATION
    # ========================================================================
    
    print(f"\n[1/6] Data preparation and validation...")
    
    # Validate input data
    if not return_series_dict:
        raise ValueError("Empty return_series_dict provided")
    
    total_strategies = len(return_series_dict)
    print(f"   ✓ Processing {total_strategies} strategy combinations")
    print(f"   ✓ Formation periods: {formation_periods}")
    print(f"   ✓ Holding periods: {holding_periods}")
    print(f"   ✓ Market state lags: {market_state_lags}")
    
    # Create aligned return matrix
    # All return series must have common date index for correlation calculation
    all_returns = pd.DataFrame(return_series_dict)
    
    # Remove any rows with all NaN values (dates not common to all strategies)
    all_returns = all_returns.dropna(how='all')
    
    # For strategies with missing values, use only overlapping periods
    valid_strategies = all_returns.columns[all_returns.notna().sum() > 50]  # Require ≥50 observations
    
    if len(valid_strategies) < total_strategies:
        print(f"   ⚠ Warning: {total_strategies - len(valid_strategies)} strategies excluded due to insufficient data")
        all_returns = all_returns[valid_strategies]
    
    print(f"   ✓ Final dataset: {len(all_returns)} months × {len(all_returns.columns)} strategies")
    
    # ========================================================================
    # STEP 2: CORRELATION MATRIX COMPUTATION
    # ========================================================================
    
    print(f"\n[2/6] Computing correlation matrix...")
    
    # Calculate pairwise correlations using only overlapping observations
    correlation_matrix = all_returns.corr(method='pearson', min_periods=30)
    
    # Handle any remaining NaN values (from strategies with no overlap)
    correlation_matrix = correlation_matrix.fillna(0)
    
    # Ensure matrix is symmetric and positive semi-definite
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix.values, 1.0)  # Ensure diagonal = 1
    
    print(f"   ✓ Correlation matrix computed: {correlation_matrix.shape}")
    print(f"   ✓ Average pairwise correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
    
    # Display correlation summary statistics
    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    print(f"   ✓ Correlation statistics:")
    print(f"      - Minimum: {corr_values.min():.3f}")
    print(f"      - Maximum: {corr_values.max():.3f}")
    print(f"      - Median: {np.median(corr_values):.3f}")
    
    # ========================================================================
    # STEP 3: EIGENVALUE DECOMPOSITION
    # ========================================================================
    
    print(f"\n[3/6] Performing eigenvalue decomposition...")
    
    # Compute eigenvalues (sorted in descending order)
    eigenvalues = eigvals(correlation_matrix.values)
    eigenvalues = np.real(eigenvalues)  # Remove tiny imaginary components from numerical errors
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    # Ensure all eigenvalues are non-negative (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Calculate variance explained by each eigenvalue
    total_variance = eigenvalues.sum()
    variance_explained = np.cumsum(eigenvalues) / total_variance
    
    print(f"   ✓ Eigenvalue decomposition complete")
    print(f"   ✓ Total variance (trace): {total_variance:.1f}")
    print(f"   ✓ Largest eigenvalue: {eigenvalues[0]:.3f}")
    print(f"   ✓ Smallest eigenvalue: {eigenvalues[-1]:.6f}")
    
    # ========================================================================
    # STEP 4: DETERMINE EFFECTIVE NUMBER OF TESTS
    # ========================================================================
    
    print(f"\n[4/6] Determining effective number of independent tests...")
    
    # Find minimum k such that first k eigenvalues explain ≥99% of variance
    # This represents the effective dimensionality of the strategy space
    variance_threshold = 0.99
    effective_tests = np.argmax(variance_explained >= variance_threshold) + 1
    
    print(f"   ✓ Variance threshold: {variance_threshold*100:.0f}%")
    print(f"   ✓ Effective number of tests (k): {effective_tests}")
    print(f"   ✓ Variance explained by first {effective_tests} components: {variance_explained[effective_tests-1]*100:.2f}%")
    print(f"   ✓ Reduction factor: {total_strategies}/{effective_tests} = {total_strategies/effective_tests:.2f}")
    
    # Display eigenvalue breakdown
    print(f"   ✓ Top 5 eigenvalues and variance explained:")
    for i in range(min(5, len(eigenvalues))):
        print(f"      {i+1}: λ={eigenvalues[i]:.3f}, cumulative={variance_explained[i]*100:.1f}%")
    
    # ========================================================================
    # STEP 5: COMPUTE CORRECTED CRITICAL VALUES
    # ========================================================================
    
    print(f"\n[5/6] Computing Bonferroni-corrected critical values...")
    
    # Standard critical value (no correction)
    raw_critical = stats.norm.ppf(1 - significance_level/2)
    
    # Bonferroni-corrected critical value using effective number of tests
    corrected_alpha = significance_level / (2 * effective_tests)
    critical_value = stats.norm.ppf(1 - corrected_alpha)
    
    print(f"   ✓ Significance level: {significance_level*100:.1f}%")
    print(f"   ✓ Raw critical t-value (no correction): ±{raw_critical:.3f}")
    print(f"   ✓ Corrected significance level: {corrected_alpha*100:.4f}%")
    print(f"   ✓ Bonferroni-corrected critical t-value: ±{critical_value:.3f}")
    print(f"   ✓ Correction magnitude: {critical_value/raw_critical:.3f}× more stringent")
    
    # ========================================================================
    # STEP 6: INDIVIDUAL STRATEGY TESTING WITH HAC-ADJUSTED T-STATISTICS
    # ========================================================================
    
    print(f"\n[6/6] Computing individual strategy statistics...")
    
    individual_results = {}
    significant_strategies = []
    raw_rejections = 0
    corrected_rejections = 0
    
    for i, (strategy_key, returns) in enumerate(return_series_dict.items()):
        print(f"   Processing strategy {i+1}/{total_strategies}: {strategy_key}", end=" ")
        
        try:
            # Extract parameters for HAC lag calculation
            market_state_lag, J, K = strategy_key
            
            # Calculate proper HAC lag structure for this strategy
            # L = K + market_state_lag - 1 (accounts for overlapping portfolios + state persistence)
            hac_lags = K + market_state_lag - 1
            
            # Clean return series
            clean_returns = returns.dropna()
            
            if len(clean_returns) < max(50, hac_lags + 10):  # Require sufficient observations
                print("→ Insufficient data")
                continue
            
            # Perform HAC-adjusted regression: R_t = α + ε_t
            # Test H₀: α = 0 (no abnormal returns)
            X = np.ones(len(clean_returns))  # Constant term only
            
            # Use Newey-West HAC standard errors
            ols = sm.OLS(clean_returns, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': hac_lags}
            )
            
            # Extract test statistics
            mean_return = ols.params[0]
            t_statistic = ols.tvalues[0]
            p_value = ols.pvalues[0]
            hac_se = ols.bse[0]
            
            # Significance tests
            raw_significant = abs(t_statistic) > raw_critical
            corrected_significant = abs(t_statistic) > critical_value
            
            # Count rejections
            if raw_significant:
                raw_rejections += 1
            if corrected_significant:
                corrected_rejections += 1
                significant_strategies.append(strategy_key)
            
            # Store detailed results
            individual_results[strategy_key] = {
                'mean_return': mean_return,
                't_statistic': t_statistic,
                'p_value': p_value,
                'hac_se': hac_se,
                'hac_lags': hac_lags,
                'n_obs': len(clean_returns),
                'raw_significant': raw_significant,
                'corrected_significant': corrected_significant
            }
            
            # Progress indicator
            significance_flag = "***" if corrected_significant else "**" if raw_significant else ""
            print(f"→ t={t_statistic:.3f}{significance_flag}")
            
        except Exception as e:
            print(f"→ Error: {str(e)[:30]}...")
            continue
    
    # ========================================================================
    # RESULTS SUMMARY AND COMPILATION
    # ========================================================================
    
    print(f"\n" + "-" * 80)
    print("BONFERRONI CORRECTION RESULTS SUMMARY")
    print("-" * 80)
    
    rejection_summary = {
        'total_strategies': total_strategies,
        'effective_tests': effective_tests,
        'raw_rejections': raw_rejections,
        'corrected_rejections': corrected_rejections,
        'raw_rejection_rate': raw_rejections / total_strategies,
        'corrected_rejection_rate': corrected_rejections / total_strategies,
        'false_discovery_reduction': raw_rejections - corrected_rejections
    }
    
    print(f"Total strategies tested: {total_strategies}")
    print(f"Effective independent tests: {effective_tests}")
    print(f"Raw significant strategies (α={significance_level}): {raw_rejections} ({raw_rejections/total_strategies*100:.1f}%)")
    print(f"Bonferroni-corrected significant: {corrected_rejections} ({corrected_rejections/total_strategies*100:.1f}%)")
    print(f"False discoveries eliminated: {raw_rejections - corrected_rejections}")
    
    if significant_strategies:
        print(f"\nStrategies significant after Bonferroni correction:")
        for strategy in significant_strategies:
            result = individual_results[strategy]
            lag, J, K = strategy
            print(f"   Market Lag={lag}, J={J}, K={K}: t={result['t_statistic']:.3f}, " +
                  f"return={result['mean_return']*100:.3f}%/month")
    else:
        print(f"\nNo strategies remain significant after Bonferroni correction.")
        print(f"This suggests that apparent significance may be due to multiple testing bias.")
    
    # ========================================================================
    # COMPILE COMPREHENSIVE RESULTS
    # ========================================================================
    
    results = {
        'correlation_matrix': correlation_matrix,
        'eigenvalues': eigenvalues,
        'variance_explained': variance_explained,
        'effective_tests': effective_tests,
        'critical_value': critical_value,
        'raw_critical': raw_critical,
        'individual_results': individual_results,
        'significant_strategies': significant_strategies,
        'rejection_summary': rejection_summary,
        'return_matrix': all_returns
    }
    
    print(f"\n" + "=" * 80)
    print("BONFERRONI CORRECTION ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


def analyze_bonferroni_results_academic(bonferroni_results, save_path_base=None):
    """
    Generate comprehensive academic visualizations for Bonferroni correction analysis.
    
    Creates publication-quality plots examining multiple testing corrections and their
    impact on strategy significance. Designed for inclusion in academic papers with
    proper statistical interpretation and professional formatting.
    
    Parameters
    ----------
    bonferroni_results : dict
        Results from compute_bonferroni_corrected_statistics()
        
    save_path_base : str, optional
        Base path for saving figures (without extension)
        
    Returns
    -------
    None
        Generates and displays/saves four publication-ready plots
    """
    
    print("\n" + "-" * 80)
    print("GENERATING ACADEMIC VISUALIZATIONS FOR BONFERRONI ANALYSIS")
    print("-" * 80)
    
    # ========================================================================
    # PLOT 1: EIGENVALUE SCREE PLOT
    # ========================================================================
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    eigenvalues = bonferroni_results['eigenvalues']
    n_eigs = len(eigenvalues)
    
    # Plot eigenvalues
    ax1.plot(range(1, n_eigs + 1), eigenvalues, 'ko-', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)
    
    # Highlight effective number of tests
    effective_k = bonferroni_results['effective_tests']
    ax1.axvline(effective_k, color='#666666', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Effective Tests (k={effective_k})')
    
    # Add 99% variance threshold annotation
    variance_99_idx = np.argmax(bonferroni_results['variance_explained'] >= 0.99) + 1
    ax1.annotate(f'99% Variance\n(k={variance_99_idx})', 
                xy=(variance_99_idx, eigenvalues[variance_99_idx-1]),
                xytext=(variance_99_idx + 5, eigenvalues[variance_99_idx-1] + 0.5),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=9, ha='left')
    
    ax1.set_title('Eigenvalue Decomposition of Strategy Correlation Matrix\n' +
                  'Determining Effective Number of Independent Tests',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Eigenvalue', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path_base:
        plt.savefig(f"{save_path_base}_eigenvalues.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_base}_eigenvalues.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # PLOT 2: CORRELATION MATRIX HEATMAP
    # ========================================================================
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create correlation heatmap
    corr_matrix = bonferroni_results['correlation_matrix']
    
    # Use diverging colormap centered at 0
    sns.heatmap(corr_matrix, 
                center=0,
                cmap='RdGy_r',  # Red-Gray colormap (reversed)
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.1,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax2)
    
    ax2.set_title('Strategy Return Correlation Matrix\n' +
                  'Motivation for Eigenvalue-Based Multiple Testing Correction',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Strategy Combination', fontsize=11)
    ax2.set_ylabel('Strategy Combination', fontsize=11)
    
    # Rotate labels for better readability
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    if save_path_base:
        plt.savefig(f"{save_path_base}_correlation.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_base}_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # PLOT 3: T-STATISTIC COMPARISON
    # ========================================================================
    
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract t-statistics
    individual_results = bonferroni_results['individual_results']
    t_stats = [result['t_statistic'] for result in individual_results.values()]
    strategy_labels = [f"({key[0]},{key[1]},{key[2]})" for key in individual_results.keys()]
    
    # Sort by t-statistic for better visualization
    sorted_indices = np.argsort(t_stats)[::-1]  # Descending order
    t_stats_sorted = [t_stats[i] for i in sorted_indices]
    
    # Plot t-statistics
    x_pos = range(len(t_stats_sorted))
    bars = ax3.bar(x_pos, t_stats_sorted, 
                   color=['#404040' if abs(t) > bonferroni_results['critical_value'] else '#808080' 
                          for t in t_stats_sorted],
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add critical value lines
    raw_critical = bonferroni_results['raw_critical']
    corrected_critical = bonferroni_results['critical_value']
    
    ax3.axhline(raw_critical, color='#666666', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Raw Critical Value (±{raw_critical:.2f})')
    ax3.axhline(-raw_critical, color='#666666', linestyle='--', linewidth=2, alpha=0.8)
    
    ax3.axhline(corrected_critical, color='black', linestyle='-', linewidth=2,
                label=f'Bonferroni Critical Value (±{corrected_critical:.2f})')
    ax3.axhline(-corrected_critical, color='black', linestyle='-', linewidth=2)
    
    ax3.set_title('t-Statistics with Multiple Testing Correction\n' +
                  f'Strategies Significant: Raw={bonferroni_results["rejection_summary"]["raw_rejections"]}, ' +
                  f'Corrected={bonferroni_results["rejection_summary"]["corrected_rejections"]}',
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel('Strategy (Ranked by t-Statistic)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('t-Statistic (HAC-Adjusted)', fontsize=11, fontweight='bold')
    
    # Legend
    ax3.legend(loc='upper right', fontsize=9)
    
    # Grid and formatting
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Remove x-tick labels due to crowding
    ax3.set_xticks([])
    
    plt.tight_layout()
    if save_path_base:
        plt.savefig(f"{save_path_base}_tstatistics.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_base}_tstatistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # PLOT 4: VARIANCE EXPLAINED CUMULATIVE
    # ========================================================================
    
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
    
    variance_explained = bonferroni_results['variance_explained']
    n_components = len(variance_explained)
    
    # Plot cumulative variance explained
    ax4.plot(range(1, n_components + 1), variance_explained * 100, 
             'ko-', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)
    
    # Add 99% threshold line
    ax4.axhline(99, color='#666666', linestyle='--', linewidth=2, alpha=0.8,
                label='99% Threshold')
    
    # Highlight effective number of tests
    effective_k = bonferroni_results['effective_tests']
    ax4.axvline(effective_k, color='black', linestyle='-', linewidth=2,
                label=f'Effective Tests (k={effective_k})')
    
    # Fill area under curve
    ax4.fill_between(range(1, n_components + 1), variance_explained * 100, 
                     alpha=0.2, color='#808080')
    
    ax4.set_title('Cumulative Variance Explained by Principal Components\n' +
                  'Basis for Effective Number of Independent Tests',
                  fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('Number of Principal Components', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Variance Explained (%)', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 100)
    
    # Legend
    ax4.legend(loc='lower right', fontsize=9)
    
    # Grid and formatting
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path_base:
        plt.savefig(f"{save_path_base}_variance.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path_base}_variance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Generated 4 academic visualizations for Bonferroni analysis")


def plot_market_dependent_academic(decile_ret, title=None, save_path=None):
    """
    Create publication-quality GRAYSCALE plot of cumulative returns with market states.
    
    This function generates a time series plot following academic journal standards:
    - Grayscale color scheme for print compatibility
    - Market state regimes shown as shaded backgrounds
    - Cumulative log returns for proper compounding visualization
    - Professional formatting suitable for journal publication
    
    Parameters
    ----------
    decile_ret : pd.DataFrame
        Portfolio returns DataFrame from analyze_momentum_portfolio()
        Must contain columns:
        - cum_log_long_short: Cumulative log returns
        - market_state: Market regime indicator (1, 0, -1)
        
    title : str, optional
        Custom plot title. If None, uses default momentum strategy title
        
    save_path : str, optional
        Path to save figure. Saves at 300 DPI for publication quality
        If provided, saves as both PDF and PNG formats
        
    Notes
    -----
    Visual Design:
    - Light gray shading: Positive market states (bull markets)
    - Dark gray shading: Negative market states (bear markets)
    - No shading: Neutral market states
    - Log scale preserves additive properties of returns
    - Black line for main series ensures maximum contrast
    
    The function automatically detects regime changes and shades
    continuous periods with the same market state as single blocks
    for cleaner visualization.
    """
    
    print("\nGenerating academic performance visualization...")
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    # Ensure datetime index for proper x-axis formatting
    decile_ret.index = pd.to_datetime(decile_ret.index)
    
    # Create figure with publication-standard dimensions
    # 8x5 inches provides good aspect ratio for journal columns
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # ========================================================================
    # PLOT CUMULATIVE RETURNS
    # ========================================================================
    
    # Main performance line
    # Use black for maximum contrast in grayscale
    ax.plot(decile_ret.index, 
            decile_ret['cum_log_long_short'], 
            color='black', 
            linewidth=1.5,  # Thick enough for clarity
            label='Cumulative Log Returns',
            zorder=3)  # Ensure line is on top
    
    # ========================================================================
    # ADD MARKET STATE SHADING
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Identify regime changes
    # ------------------------------------------------------------------------
    # Create groups for continuous periods of the same state
    # This prevents overlapping rectangles and improves rendering
    decile_ret['state_change'] = decile_ret['market_state'].ne(
        decile_ret['market_state'].shift()
    ).cumsum()
    
    # ------------------------------------------------------------------------
    # Color each regime period
    # ------------------------------------------------------------------------
    # Group by regime and state to get continuous periods
    for (change_group, state), group in decile_ret.groupby(['state_change', 'market_state']):
        if len(group) > 0 and not pd.isna(state):
            # Get regime boundaries
            start_date = group.index[0]
            end_date = group.index[-1]
            
            # Determine color based on market state (grayscale)
            if state == 1:
                color = '#E0E0E0'  # Light gray for positive state
                label = 'Positive Market State'
            elif state == -1:
                color = '#808080'  # Medium gray for negative state
                label = 'Negative Market State'
            else:
                continue  # Skip neutral states (no shading)
            
            # Add vertical span for regime
            ax.axvspan(start_date, end_date, 
                      alpha=0.3,  # Semi-transparent
                      color=color,
                      zorder=1)  # Behind the main line
    
    # ========================================================================
    # FORMATTING FOR PUBLICATION
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # Title and axis labels
    # ------------------------------------------------------------------------
    if title is None:
        # Default title emphasizes academic nature
        title = 'Market-State Dependent Momentum Strategy: Cumulative Performance'
    
    ax.set_title(title, 
                fontsize=12, 
                fontweight='bold', 
                pad=15)
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Log Return', fontsize=11, fontweight='bold')
    
    # ------------------------------------------------------------------------
    # X-axis date formatting
    # ------------------------------------------------------------------------
    # Major ticks every 10 years for long time series
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Minor ticks every 5 years for additional reference
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))
    
    # ------------------------------------------------------------------------
    # Grid and reference lines
    # ------------------------------------------------------------------------
    # Subtle grid improves readability without distraction
    ax.grid(True, 
            alpha=0.2,          # Very light grid
            linestyle='-',      # Solid lines
            linewidth=0.5,      # Thin lines
            color='gray')       # Gray color
    
    ax.set_axisbelow(True)  # Ensure grid renders behind data
    
    # Add zero line for reference (break-even point)
    ax.axhline(y=0, 
              color='black', 
              linestyle='-', 
              linewidth=0.8, 
              alpha=0.5)
    
    # ------------------------------------------------------------------------
    # Legend with market state explanation
    # ------------------------------------------------------------------------
    # Create custom legend elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, 
               label='Cumulative Log Returns'),
        Patch(facecolor='#E0E0E0', alpha=0.3, 
              label='UP Market State'),
        Patch(facecolor='#808080', alpha=0.3, 
              label='DOWN Market State')
    ]
    
    # Position legend with academic styling
    ax.legend(handles=legend_elements, 
              loc='upper left',
              frameon=True,          # Box around legend
              fancybox=False,        # Simple rectangle
              shadow=False,          # No shadow for print
              fontsize=9,
              edgecolor='black',     # Black border
              facecolor='white',     # White background
              framealpha=0.8)        # Slight transparency
    
    # ------------------------------------------------------------------------
    # Additional formatting
    # ------------------------------------------------------------------------
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines slightly thicker
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # ------------------------------------------------------------------------
    # Final layout adjustments
    # ------------------------------------------------------------------------
    plt.tight_layout()
    
    # ========================================================================
    # SAVE AND DISPLAY
    # ========================================================================
    
    if save_path:
        # Save in multiple formats for flexibility
        # PDF for LaTeX inclusion
        plt.savefig(f"{save_path}.pdf", 
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        # PNG for other uses
        plt.savefig(f"{save_path}.png", 
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        print(f"   ✓ Figure saved as: {save_path}.pdf and {save_path}.png")
    
    plt.show()
    print("   ✓ Visualization complete")


def print_portfolio_summary(analysis_results, strategy_name="Momentum Strategy"):
    """
    Print formatted portfolio performance summary for academic reporting.
    
    Creates a well-formatted table suitable for inclusion in academic papers
    or presentations, with automatic significance level indicators following
    standard academic conventions.
    
    Parameters
    ----------
    analysis_results : dict
        Output dictionary from analyze_momentum_portfolio() containing:
        - avg_monthly_return: Average monthly return
        - monthly_vol: Monthly volatility
        - sharpe_ratio: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown
        - t_stat: HAC-adjusted t-statistic
        - p_value: Two-sided p-value
        
    strategy_name : str, default="Momentum Strategy"
        Descriptive name for the strategy being analyzed
        
    Notes
    -----
    Significance levels follow standard academic conventions:
    - *** : p < 0.01 (1% level, highly significant)
    - **  : p < 0.05 (5% level, significant)
    - *   : p < 0.10 (10% level, marginally significant)
    
    The summary is formatted for easy inclusion in LaTeX documents
    or presentation slides.
    """
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    
    # Create formatted header with consistent width
    print(f"\n{'=' * 65}")
    print(f"{strategy_name:^65}")            # Center strategy name
    print(f"{'Performance Summary':^65}")     # Center subtitle
    print(f"{'=' * 65}")
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    # Format each metric with consistent alignment and precision
    # All metrics right-aligned at column 30 for clean appearance
    
    # Return and risk metrics (as percentages)
    print(f"Average Monthly Return:      {analysis_results['avg_monthly_return']:>8.2%}")
    print(f"Monthly Volatility:          {analysis_results['monthly_vol']:>8.2%}")
    
    # Risk-adjusted performance
    print(f"Annualized Sharpe Ratio:     {analysis_results['sharpe_ratio']:>8.3f}")
    
    # Downside risk
    print(f"Maximum Drawdown:            {analysis_results['max_drawdown']:>8.2%}")
    
    # Statistical significance
    print(f"t-statistic (HAC-adjusted):  {analysis_results['t_stat']:>8.4f}")
    print(f"p-value (two-sided):         {analysis_results['p_value']:>8.4e}")
    
    # ========================================================================
    # SIGNIFICANCE INDICATORS
    # ========================================================================
    
    # Determine significance level using academic conventions
    if analysis_results['p_value'] < 0.01:
        sig = "***"  # Highly significant (1% level)
        sig_text = "Highly significant (p < 0.01)"
    elif analysis_results['p_value'] < 0.05:
        sig = "**"   # Significant (5% level)
        sig_text = "Significant (p < 0.05)"
    elif analysis_results['p_value'] < 0.10:
        sig = "*"    # Marginally significant (10% level)
        sig_text = "Marginally significant (p < 0.10)"
    else:
        sig = ""     # Not significant
        sig_text = "Not significant"
    
    # Print significance with description
    print(f"Statistical Significance:    {sig:>8} {sig_text}")
    
    # ========================================================================
    # INTERPRETATION SECTION
    # ========================================================================
    
    print(f"\nInterpretation:")
    
    # Sharpe ratio interpretation
    if analysis_results['sharpe_ratio'] > 2:
        sharpe_desc = "Exceptional"
    elif analysis_results['sharpe_ratio'] > 1:
        sharpe_desc = "Very good"
    elif analysis_results['sharpe_ratio'] > 0.5:
        sharpe_desc = "Good"
    else:
        sharpe_desc = "Poor"
    
    print(f"  - Sharpe Ratio: {sharpe_desc} risk-adjusted performance")
    
    # Maximum drawdown interpretation
    if abs(analysis_results['max_drawdown']) < 0.10:
        dd_desc = "Low risk"
    elif abs(analysis_results['max_drawdown']) < 0.20:
        dd_desc = "Moderate risk"
    else:
        dd_desc = "High risk"
    
    print(f"  - Maximum Drawdown: {dd_desc} profile")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    print(f"{'=' * 65}\n")


def compute_df_t_stat_and_avg_return(
    dataframe: pd.DataFrame,
    window_months: int = 360,
    lags: int = 2,
    date_col: str = 'date',
    return_col: str = 'long_short',
    min_obs_ratio: float = 0.8
) -> pd.DataFrame:
    """
    Compute rolling window HAC-adjusted t-statistics and performance metrics.
    
    This function performs rolling window analysis with proper Newey-West
    standard errors to account for serial correlation in overlapping portfolios.
    Essential for examining strategy stability and regime dependence over time.
    
    IMPORTANT: The 'lags' parameter should be set to K + market_state_lag - 1
    for market state-dependent strategies.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Time series data containing returns and dates
        Should have datetime index or date column
        
    window_months : int, default=360
        Rolling window size in months
        Common choices: 
        - 60 (5yr): Short-term stability
        - 120 (10yr): Medium-term patterns
        - 240 (20yr): Long-term consistency
        - 360 (30yr): Full cycle analysis
        
    lags : int, default=2
        Number of lags for HAC standard errors
        For market state strategies: K + market_state_lag - 1
        Accounts for autocorrelation from overlapping portfolios and state persistence
        
    date_col : str, default='date'
        Name of the date column in dataframe
        
    return_col : str, default='long_short'
        Name of the return column to analyze
        
    min_obs_ratio : float, default=0.8
        Minimum fraction of non-missing observations required
        E.g., 0.8 requires at least 80% of window_months observations
        Prevents unreliable estimates from sparse data
        
    Returns
    -------
    pd.DataFrame
        Rolling window statistics with columns:
        - Start Date: Window start date
        - End Date: Window end date (inclusive)
        - Average Return: Mean return in window
        - T-stat: HAC-adjusted t-statistic
        - P-value: Two-sided p-value
        - Significant: Boolean indicator (p < 0.05)
        - N_Obs: Number of observations used
        - Volatility: Return standard deviation
        - Sharpe_Ratio: Annualized Sharpe ratio
        
    Notes
    -----
    Implementation Details:
    - Windows advance by one month (non-overlapping starts)
    - Each window must have minimum observations to be included
    - HAC standard errors use Newey-West with specified lag structure
    - Sharpe ratios assume zero risk-free rate
    
    The function is useful for:
    - Detecting performance persistence over time
    - Identifying regime-dependent returns
    - Validating out-of-sample strategy robustness
    - Understanding time-varying risk premia
    """
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    # Create working copy and ensure proper date formatting
    df = dataframe.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Determine data range
    last_date = df[date_col].max()
    
    # Initialize results storage
    results = []
    start_idx = 0
    
    # ========================================================================
    # ROLLING WINDOW LOOP
    # ========================================================================
    
    print(f"Computing {window_months}-month rolling window statistics...")
    print(f"   Using {lags} lags for HAC adjustment")
    
    while True:
        # ------------------------------------------------------------------------
        # Define window boundaries
        # ------------------------------------------------------------------------
        win_start = df.loc[start_idx, date_col]
        win_end = win_start + pd.DateOffset(months=window_months)
        
        # Stop if complete window doesn't fit in available data
        if win_end > last_date:
            break
        
        # ------------------------------------------------------------------------
        # Extract window data
        # ------------------------------------------------------------------------
        # Select observations within window
        window_mask = (df[date_col] >= win_start) & (df[date_col] < win_end)
        y = df.loc[window_mask, return_col].dropna().values
        
        # ------------------------------------------------------------------------
        # Check minimum observations requirement
        # ------------------------------------------------------------------------
        # Require at least min_obs_ratio of potential observations
        if len(y) >= window_months * min_obs_ratio:
            # ----------------------------------------------------------------
            # Calculate basic statistics
            # ----------------------------------------------------------------
            avg_return = np.mean(y)
            volatility = np.std(y, ddof=1)  # Sample standard deviation
            
            # Annualized Sharpe ratio (assuming Rf = 0)
            sharpe_ratio = (avg_return / volatility) * np.sqrt(12) if volatility > 0 else np.nan
            
            # ----------------------------------------------------------------
            # HAC-adjusted statistical testing
            # ----------------------------------------------------------------
            try:
                # OLS regression on constant (testing mean return)
                # Model: y_t = α + ε_t, where α is the mean return
                ols = sm.OLS(y, np.ones_like(y)).fit(
                    cov_type='HAC',                    # Newey-West errors
                    cov_kwds={'maxlags': lags}         # Extended lag structure
                )
                
                t_stat = ols.tvalues[0]    # t-stat for constant term
                p_value = ols.pvalues[0]   # Two-sided p-value
            except:
                # Handle potential estimation failures
                # Can occur with extreme data or numerical issues
                t_stat = np.nan
                p_value = np.nan
            
            # ----------------------------------------------------------------
            # Store results
            # ----------------------------------------------------------------
            results.append({
                'Start Date': win_start,
                'End Date': win_end - pd.DateOffset(days=1),  # Inclusive end
                'Average Return': avg_return,
                'T-stat': t_stat,
                'P-value': p_value,
                'Significant': p_value < 0.05 if not np.isnan(p_value) else False,
                'N_Obs': len(y),
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio
            })
        
        # ------------------------------------------------------------------------
        # Advance to next window
        # ------------------------------------------------------------------------
        # Move forward by one month
        next_month = win_start + pd.DateOffset(months=1)
        next_indices = df.index[df[date_col] >= next_month]
        
        # Check if more data available
        if len(next_indices) == 0:
            break
            
        start_idx = next_indices[0]
    
    # ========================================================================
    # RETURN RESULTS
    # ========================================================================
    
    print(f"   ✓ Computed {len(results)} rolling windows")
    
    return pd.DataFrame(results)


def plot_tstat_and_avg_return_academic(results_df: pd.DataFrame, 
                                       title_suffix: str = "",
                                       save_path: str = None,
                                       window_months: int = 240) -> None:
    """
    Create publication-quality GRAYSCALE visualizations of rolling-window statistics.
    
    Generates three separate academic-style plots suitable for journal publication:
    1. HAC-adjusted t-statistics with significance bands
    2. Average returns with confidence intervals
    3. Rolling Sharpe ratios with reference levels
    
    Each plot uses grayscale design for print compatibility and follows
    standard academic formatting conventions.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from compute_df_t_stat_and_avg_return() containing columns:
        ['Start Date', 'T-stat', 'Significant', 'Average Return',
         'Volatility', 'N_Obs', 'Sharpe_Ratio']
         
    title_suffix : str, default=""
        Additional text appended to plot titles
        E.g., "(1993-2023)" or "Market-State Adjusted"
        
    save_path : str, optional
        Base path for saving plots. If provided, saves six files:
        - {save_path}_tstat.pdf and .png
        - {save_path}_returns.pdf and .png
        - {save_path}_sharpe.pdf and .png
        
    window_months : int, default=240
        Number of months in rolling window (for x-axis calculation)
        
    Notes
    -----
    Design Philosophy:
    - Grayscale palette for journal compatibility
    - Significance bands show contiguous periods as single shaded regions
    - Professional formatting with serif fonts
    - Grid lines for easy value reading
    - All fonts sized for readability in print
    
    Statistical Visualization:
    - T-statistics show ±1.96 (5%) and ±2.58 (1%) significance levels
    - Returns include 95% confidence intervals when volatility available
    - Sharpe ratios show 0.5, 1.0, and 1.5 reference levels
    
    The plots are designed to be included directly in academic papers
    without further modification.
    """
    
    # ========================================================================
    # DATA VALIDATION
    # ========================================================================
    
    if results_df.empty:
        print("No data to plot - results DataFrame is empty")
        return

    # ------------------------------------------------------------------------
    # Prepare x-axis data (window end dates)
    # ------------------------------------------------------------------------
    # Use end dates for x-axis as they represent when results are known
    start_dates = pd.to_datetime(results_df['Start Date'])
    x = start_dates + pd.DateOffset(months=window_months)  # End dates
    
    # Extract significance indicator
    sig = results_df['Significant'].fillna(False).astype(bool).values

    # ========================================================================
    # PLOT 1: HAC t-statistics (Grayscale)
    # ========================================================================
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    # ------------------------------------------------------------------------
    # Main t-statistic line
    # ------------------------------------------------------------------------
    ax1.plot(x, results_df['T-stat'],
             color='black',           # Primary data in black
             linewidth=1.5,           # Thick for emphasis
             label='HAC t-statistic',
             zorder=3)               # On top

    # ------------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------------
    # Zero line (null hypothesis)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Significance thresholds
    for y, label in [(2.58, '±2.58 (1% level)'), 
                     (-2.58, None),
                     (1.96, '±1.96 (5% level)'), 
                     (-1.96, None)]:
        ax1.axhline(y=y, 
                   color='#666666',    # Dark gray
                   linestyle='--' if abs(y) == 1.96 else ':',
                   linewidth=1.0, 
                   alpha=0.7,
                   label=label)

    # ------------------------------------------------------------------------
    # Identify and shade significant periods
    # ------------------------------------------------------------------------
    # Collapse contiguous runs of significance into single spans
    runs = []
    in_run = False
    run_start = None

    for i, is_sig in enumerate(sig):
        if is_sig and not in_run:
            # Start of significant period
            in_run = True
            run_start = x[i]
        elif in_run and (not is_sig or i == len(sig) - 1):
            # End of significant period
            run_end = x[i] if sig[i] else x[i-1]
            runs.append((run_start, run_end))
            in_run = False

    # Add shading for significant periods
    for start, end in runs:
        ax1.axvspan(start, end, 
                   color='#D0D0D0',     # Light gray
                   alpha=0.3,           # Semi-transparent
                   label='Significant periods' if start == runs[0][0] else "",
                   zorder=1)

    # ------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------
    ax1.set_title(f"Rolling {window_months//12}-Year t-Statistics {title_suffix}",
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Window End Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('t-statistic (HAC-adjusted)', fontsize=11, fontweight='bold')
    
    # Professional legend
    ax1.legend(loc='best', 
              fontsize=9, 
              frameon=True,
              edgecolor='black',
              facecolor='white',
              framealpha=0.9)
    
    # Grid
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(f"{save_path}_tstat.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_tstat.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # PLOT 2: Rolling Average Monthly Returns (Grayscale)
    # ========================================================================
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    
    # Convert returns to percentage for readability
    returns_pct = results_df['Average Return'] * 100
    
    # ------------------------------------------------------------------------
    # Main returns line
    # ------------------------------------------------------------------------
    ax2.plot(x, returns_pct,
             color='black', 
             linewidth=1.5, 
             label='Average Monthly Return',
             zorder=3)
    
    # Zero line
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # ------------------------------------------------------------------------
    # Confidence intervals (if volatility available)
    # ------------------------------------------------------------------------
    if 'Volatility' in results_df.columns and 'N_Obs' in results_df.columns:
        # Calculate 95% confidence interval
        # CI = mean ± 1.96 * SE, where SE = σ/√n
        se = results_df['Volatility'] / np.sqrt(results_df['N_Obs']) * 100
        upper = returns_pct + 1.96 * se
        lower = returns_pct - 1.96 * se
        
        # Shade confidence interval
        ax2.fill_between(x, lower, upper,
                         alpha=0.2, 
                         color='#808080',        # Medium gray
                         label='95% Confidence Interval',
                         zorder=2)

    # ------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------
    ax2.set_title(f"Rolling {window_months//12}-Year Average Returns {title_suffix}",
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Window End Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Monthly Return (%)', fontsize=11, fontweight='bold')
    
    # Legend
    ax2.legend(loc='best', 
              fontsize=9, 
              frameon=True,
              edgecolor='black',
              facecolor='white',
              framealpha=0.9)
    
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_returns.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_returns.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # PLOT 3: Rolling Sharpe Ratio (Grayscale)
    # ========================================================================
    
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    
    # ------------------------------------------------------------------------
    # Calculate and plot Sharpe ratio
    # ------------------------------------------------------------------------
    if 'Sharpe_Ratio' in results_df.columns:
        sharpe_ratio = results_df['Sharpe_Ratio'].fillna(0)
    else:
        print("Warning: Sharpe_Ratio column not found")
        return
    
    # Main Sharpe ratio line
    ax3.plot(x, sharpe_ratio,
             color='black', 
             linewidth=1.5, 
             label='Rolling Sharpe Ratio',
             zorder=3)
    
    # ------------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------------
    # Zero line
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Sharpe ratio benchmarks
    for level, style, label in [(1.5, ':', 'SR = 1.5 '),
                                (1.0, '--', 'SR = 1.0 '),
                                (0.5, '-.', 'SR = 0.5 ')]:
        ax3.axhline(level, 
                   color='#666666',     # Dark gray
                   linestyle=style, 
                   alpha=0.7,
                   linewidth=1.0, 
                   label=label)

    # ------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------
    ax3.set_title(f"Rolling {window_months//12}-Year Sharpe Ratio {title_suffix}",
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel('Window End Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio (Annualized)', fontsize=11, fontweight='bold')
    
    # Legend
    ax3.legend(loc='best', 
              fontsize=9, 
              frameon=True,
              edgecolor='black',
              facecolor='white',
              framealpha=0.9)
    
    ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_sharpe.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_sharpe.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # SUMMARY OUTPUT
    # ========================================================================
    
    print(f"\nGenerated academic plots:")
    if save_path:
        print(f"  - t-statistics: {save_path}_tstat.pdf/.png")
        print(f"  - Returns: {save_path}_returns.pdf/.png") 
        print(f"  - Sharpe ratio: {save_path}_sharpe.pdf/.png")