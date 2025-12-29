import pandas as pd
import numpy as np

def create_credit_data(seed=42):
    np.random.seed(seed)
    n = 2500

    data = pd.DataFrame({
        'company': [f'Corp_{i:04d}' for i in range(n)],
        'industry': np.random.choice(
            ['Manufacturing','Real Estate','Trading','Services','Construction'], n),
        'loan_amount_cr': np.random.lognormal(6, 1.8, n),
        'Beneish_M': np.random.normal(-2.45, 0.65, n),
        'Altman_Z': np.random.normal(2.9, 1.1, n),
        'Sloan_Ratio': np.random.normal(0.012, 0.025, n),
        'Piotroski_F': np.random.randint(0, 9, n),
        'DSR': np.random.uniform(0.5, 4.0, n),
        'Current_Ratio': np.random.uniform(0.5, 3.0, n),
        'Working_Cap_Days': np.random.uniform(30, 180, n),
        'Auditor_Qualification': np.random.choice([0,1], n, p=[0.85,0.15])
    })

    data['fraud_flag'] = (
        (data['Beneish_M'] > -2.22) |
        (data['Altman_Z'] < 1.81) |
        (data['Sloan_Ratio'].abs() > 0.02) |
        (data['Piotroski_F'] < 3) |
        (data['DSR'] < 1.2) |
        (data['Auditor_Qualification'] == 1)
    ).astype(int)

    return data
