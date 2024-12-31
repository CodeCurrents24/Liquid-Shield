from Extraction import *
from DataStructure	import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
def train_liquidity_model(df):
    """
    Train a predictive model for 'future_line_liquidity'
    using the engineered features.
    """
    # List your feature columns
    features = [
        'price',
        'pos_liquidity',
        'tick',
        'line_liquidity',
        'pool_price_info',
        'pool_tvl_info',
        'day_volume_info',
        'day_apr_info',
        'price_diff',
        'pos_liquidity_diff',
        'liquidity_ratio'
    ]
    
    # Target column
    target = 'future_line_liquidity'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse:.4f}")
    
    return model


def main():
    pool_id = "8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj"
    df = process_pool_data(pool_id)
    
    print("Sample of engineered data:")
    print(len(df))
    
    model = train_liquidity_model(df)
    print("Model training complete!")

if __name__ == "__main__":
    main()
