import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    
    temperature = np.random.uniform(15, 40, n_samples)  # °C
    humidity = np.random.uniform(30, 95, n_samples)     # %
    pressure = np.random.uniform(980, 1030, n_samples)  # hPa
    wind_speed = np.random.uniform(0, 60, n_samples)    # km/h
    cape_index = np.random.uniform(0, 3500, n_samples)  # J/kg
    
    # Target logic: Higher CAPE, higher humidity, higher temp increase thunderstorm probability
    score = (
        0.002 * cape_index + 
        0.05 * humidity + 
        0.08 * temperature + 
        0.03 * wind_speed - 
        0.05 * (pressure - 1000)
    )
    prob = 1 / (1 + np.exp(-(score - 8)))
    thunderstorm = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'cape_index': cape_index,
        'thunderstorm': thunderstorm
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/thunderstorm_data.csv", index=False)
    print("Data generated successfully!")