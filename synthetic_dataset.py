import pandas as pd
import numpy as np

# Number of synthetic samples
N = 1000
cbf_amplitude = {
    'tile_cbf_hz': {
        'Healthy': {'mu': 12.5, 'sigma': 1.5},
        'PCD': {'mu': 7, 'sigma': 2}
    },
    'tile_peak_amp': {
        'Healthy': {'mu': 6.5, 'sigma': 1},
        'PCD': {'mu': 5, 'sigma': 1.5}
    },
    'tile_var': {
        'Healthy': {'mu': 0.018, 'sigma': 0.005},
        'PCD': {'mu': 0.025, 'sigma': 0.008}
    },
    'tile_cbp_sine_score': {
        'Healthy': {'mu': 0.1, 'sigma': 0.05},
        'PCD': {'mu': -0.1, 'sigma': 0.05}
    },
    'tile_cbp_amp': {
        'Healthy': {'mu': 0.13, 'sigma': 0.02},
        'PCD': {'mu': 0.10, 'sigma': 0.03}
    }
}

uniform_features = {
    'tile_zcr': {
        'Healthy': {'min': 0.5, 'max': 0.7},
        'PCD': {'min': 0.6, 'max': 0.8}
    },
    'x': {
        'Healthy': {'min': 0, 'max': 512},  # frame width
        'PCD': {'min': 0, 'max': 512}
    },
    'y': {
        'Healthy': {'min': 0, 'max': 512},  # frame height
        'PCD': {'min': 0, 'max': 512}
    }
}
classes = np.random.choice(['Healthy', 'PCD'], size=N, p=[0.5, 0.5])

tile_cbf_hz = np.array([np.random.normal(cbf_amplitude['tile_cbf_hz'][c]['mu'],
                                        cbf_amplitude['tile_cbf_hz'][c]['sigma']) 
                        for c in classes])
tile_peak_amp = np.array([np.random.normal(cbf_amplitude['tile_peak_amp'][c]['mu'],
                                        cbf_amplitude['tile_peak_amp'][c]['sigma']) 
                        for c in classes])
tile_var = np.array([np.random.normal(cbf_amplitude['tile_var'][c]['mu'],
                                    cbf_amplitude['tile_var'][c]['sigma']) 
                    for c in classes])
tile_cbp_sine_score = np.array([np.random.normal(cbf_amplitude['tile_cbp_sine_score'][c]['mu'],
                                                cbf_amplitude['tile_cbp_sine_score'][c]['sigma']) 
                                for c in classes])
tile_cbp_amp = np.array([np.random.normal(cbf_amplitude['tile_cbp_amp'][c]['mu'],
                                    cbf_amplitude['tile_cbp_amp'][c]['sigma']) 
                        for c in classes])

# --- Generate uniform features ---
tile_zcr = np.array([np.random.uniform(uniform_features['tile_zcr'][c]['min'],
                                    uniform_features['tile_zcr'][c]['max']) 
                    for c in classes])
x = np.array([np.random.uniform(uniform_features['x'][c]['min'],
                            uniform_features['x'][c]['max']) 
            for c in classes])
y = np.array([np.random.uniform(uniform_features['y'][c]['min'],
                            uniform_features['y'][c]['max']) 
            for c in classes])

# --- Combine into DataFrame ---
df = pd.DataFrame({
    'tile_cbf_hz': tile_cbf_hz,
    'tile_peak_amp': tile_peak_amp,
    'tile_var': tile_var,
    'tile_zcr': tile_zcr,
    'tile_cbp_sine_score': tile_cbp_sine_score,
    'tile_cbp_amp': tile_cbp_amp,
    'x': x,
    'y': y,
    'label': classes,
    'video': ['synthetic_video.avi']*N,
    'fps': [60]*N
})
df.to_csv('synthetic_cilia_data.csv', index=False)

print("Synthetic dataset generated with", N, "rows!")
