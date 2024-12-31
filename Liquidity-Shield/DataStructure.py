from Extraction import *

import pandas as pd
import numpy as np
def process_pool_data(pool_id):
    """
    Combines data from three endpoints:
      1) 'line/position'
      2) 'line/liquidity'
      3) 'info/ids'
    into a single DataFrame with engineered features.
    
    Returns a DataFrame that can be used for modeling.
    """
    
    # Fetch raw data
    position_data = fetch_line_position(pool_id)    # list of dicts with 'price', 'liquidity', 'tick'
    liquidity_data = fetch_line_liquidity(pool_id)  # list of dicts with 'time', 'liquidity'
    info_data = fetch_pool_info(pool_id)            # dict with 'price', 'tvl', 'day', 'week', 'month', etc.
    
    # Convert the first two data sets into DataFrames
    df_position = pd.DataFrame(position_data)
    df_liquidity = pd.DataFrame(liquidity_data)
    
    # Rename for clarity
    # 'position' data doesn't include time, so you may want to treat each row as sequential or
    # see if there's a timestamp in the real data (the example snippet only shows price/liquidity/tick).
    df_position.rename(columns={'liquidity': 'pos_liquidity'}, inplace=True)
    
    # Make sure numeric columns are converted properly
    df_position['price'] = pd.to_numeric(df_position['price'], errors='coerce')
    df_position['pos_liquidity'] = pd.to_numeric(df_position['pos_liquidity'], errors='coerce')
    df_position['tick'] = pd.to_numeric(df_position['tick'], errors='coerce')
    
    # Convert time to datetime if needed
    # For 'line/liquidity', each entry has 'time' in (likely) seconds. Convert to datetime:
    df_liquidity['datetime'] = pd.to_datetime(df_liquidity['time'], unit='s')
    df_liquidity.rename(columns={'liquidity': 'line_liquidity'}, inplace=True)
    df_liquidity['line_liquidity'] = pd.to_numeric(df_liquidity['line_liquidity'], errors='coerce')
    
    # Info data is a single dict, so we can store relevant fields as constants or a small DataFrame
    pool_price = info_data.get('price', np.nan)   # e.g. 189.00918170199847
    pool_tvl   = info_data.get('tvl', np.nan)     # e.g. 8069111.49
    
    # If we want daily volume info:
    day_info   = info_data.get('day', {})
    day_volume = day_info.get('volume', np.nan)   # e.g. 216268997.2946721
    day_apr    = day_info.get('apr', np.nan)      # e.g. 99.71087211701757
    
    # For a minimal demonstration, let’s combine df_position and df_liquidity by index
    # (since they don't share a common time index in the example).
    # In a real scenario, you'd align them by a time key if available.
    
    # We'll assume df_position is high-frequency and df_liquidity is daily or some period.
    # One approach: add a row_id or time_id to df_position if the data is sequential
    df_position['row_id'] = range(len(df_position))
    df_liquidity['row_id'] = range(len(df_liquidity))
    
    # Merge on row_id as a naive demonstration:
    merged_df = pd.merge(df_position, df_liquidity, on='row_id', how='outer')
    
    # Fill in pool-level info as columns in every row
    merged_df['pool_price_info'] = pool_price
    merged_df['pool_tvl_info'] = pool_tvl
    merged_df['day_volume_info'] = day_volume
    merged_df['day_apr_info'] = day_apr
    
    # Sort by row_id or by datetime if you prefer
    merged_df.sort_values('row_id', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Now let's create some derived features
    # Example: price diff, liquidity diff, ratio, etc.
    merged_df['price_diff'] = merged_df['price'].diff()
    merged_df['pos_liquidity_diff'] = merged_df['pos_liquidity'].diff()
    merged_df['liquidity_ratio'] = merged_df['pos_liquidity'] / (merged_df['line_liquidity'] + 1e-9)
    
    # We can make a target variable: next day's line_liquidity or future price
    # For example, let's predict future line_liquidity (shifting -1)
    merged_df['future_line_liquidity'] = merged_df['line_liquidity'].shift(-1)
    
    # Drop rows with missing values caused by shifting or missing data
    merged_df.dropna(inplace=True)
    
    return merged_df
