import os
import pandas as pd
import numpy as np

def generate_thunderstorm_dataset(n_samples=2500, seed=42):
    np.random.seed(seed)
    
    # Atmospheric metrics
    temperature = np.random.uniform(18.0, 42.0, n_samples)          # °C
    dew_point = temperature - np.random.uniform(1.0, 15.0, n_samples) # °C
    relative_humidity = 100 * (np.exp((17.625 * dew_point) / (243.04 + dew_point)) / 
                               np.exp((17.625 * temperature) / (243.04 + temperature)))
    surface_pressure = np.random.uniform(970.0, 1025.0, n_samples)  # hPa
    wind_speed = np.random.uniform(2.0, 65.0, n_samples)            # km/h
    wind_shear = np.random.uniform(5.0, 45.0, n_samples)            # m/s
    cape_index = np.random.uniform(0.0, 4000.0, n_samples)          # J/kg
    k_index = np.random.uniform(5.0, 45.0, n_samples)               # Stability metric
    
    # Ground-truth instability indicator
    instability_score = (
        0.0025 * cape_index +
        0.08 * relative_humidity +
        0.09 * k_index +
        0.05 * wind_shear +
        0.06 * (temperature - 20) -
        0.04 * (surface_pressure - 1000)
    )
    
    # Sigmoid function for probability
    prob = 1 / (1 + np.exp(-(instability_score - 9.5)))
    thunderstorm_event = (prob > 0.48).astype(int)
    
    df = pd.DataFrame({
        'temperature': np.round(temperature, 2),
        'dew_point': np.round(dew_point, 2),
        'relative_humidity': np.round(relative_humidity, 2),
        'surface_pressure': np.round(surface_pressure, 2),
        'wind_speed': np.round(wind_speed, 2),
        'wind_shear': np.round(wind_shear, 2),
        'cape_index': np.round(cape_index, 2),
        'k_index': np.round(k_index, 2),
        'thunderstorm_event': thunderstorm_event
    })
    
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', 'thunderstorm_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully created at '{file_path}' with {n_samples} records.")

if __name__ == "__main__":
    generate_thunderstorm_dataset()