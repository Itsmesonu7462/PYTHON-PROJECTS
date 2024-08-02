import pandas as pd
import numpy as np

np.random.seed(42)
num_samples = 1000

data = {
    'LotArea': np.random.randint(500, 20000, num_samples),
    'OverallQual': np.random.randint(1, 10, num_samples),
    'OverallCond': np.random.randint(1, 10, num_samples),
    'YearBuilt': np.random.randint(1900, 2021, num_samples),
    'GrLivArea': np.random.randint(500, 5000, num_samples),
    'FullBath': np.random.randint(1, 4, num_samples),
    'BedroomAbvGr': np.random.randint(1, 6, num_samples),
    'TotRmsAbvGrd': np.random.randint(2, 14, num_samples),
    'GarageCars': np.random.randint(0, 4, num_samples),
    'SalePrice': np.random.randint(50000, 500000, num_samples)
}
df = pd.DataFrame(data)
df.to_csv('house_price_dataset.csv', index=True)
