import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Dew point depression (indicator of low-level moisture saturation)
    df['dew_point_depression'] = df['temperature'] - df['dew_point']
    
    # Combined Instability Metric (CAPE x Wind Shear interaction)
    df['cape_shear_product'] = df['cape_index'] * df['wind_shear']
    
    # Relative humidity bucket flag
    df['high_humidity_flag'] = (df['relative_humidity'] > 75).astype(int)
    
    return df