import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def fetch_line_position(pool_id):
    """
    Fetch data from 'pools/line/position' endpoint, which returns a list
    of price, liquidity, tick over time.
    """
    url = f"https://api-v3.raydium.io/pools/line/position?id={pool_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get("data", {})
    line_data = data.get("line", [])
    return line_data

def fetch_line_liquidity(pool_id):
    """
    Fetch data from 'pools/line/liquidity' endpoint, which returns time-based
    liquidity data.
    """
    url = f"https://api-v3.raydium.io/pools/line/liquidity?id={pool_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get("data", {})
    line_data = data.get("line", [])
    return line_data

def fetch_pool_info(pool_id):
    """
    Fetch data from 'pools/info/ids?ids=' endpoint, which returns high-level
    info like price, TVL, etc. for the pool.
    """
    url = f"https://api-v3.raydium.io/pools/info/ids?ids={pool_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[0] if len(data) > 0 else {}
