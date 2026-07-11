"""
Synthetic FoodHub order data generator.
Mimics the schema/shape of the real foodhub_order.csv (1898 rows x 9 cols)
used in the FoodHub Data Analysis notebook, WITHOUT using any real data.
Run this to produce a dummy foodhub_order.csv that the notebook can load.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1898

cuisines = ['American','Chinese','Italian','Mexican','Indian','Japanese',
            'Thai','French','Middle Eastern','Mediterranean','Southern',
            'Korean','Vietnamese','Spanish']
cuisine_probs = np.array([18,15,13,10,9,8,6,5,4,4,3,2,2,1], dtype=float)
cuisine_probs /= cuisine_probs.sum()

restaurant_pool = [f"Restaurant_{i}" for i in range(1, 180)]
# a few "popular" restaurants get much higher weight, like a real long-tail distribution
popularity = np.random.pareto(a=1.5, size=len(restaurant_pool)) + 0.1
popularity /= popularity.sum()

customer_pool = np.arange(100001, 101201)  # ~1200 unique customers, repeat customers allowed

days = np.random.choice(['Weekday', 'Weekend'], size=N, p=[0.71, 0.29])

df = pd.DataFrame({
    'order_id': np.arange(1477147, 1477147 + N),
    'customer_id': np.random.choice(customer_pool, size=N, replace=True),
    'restaurant_name': np.random.choice(restaurant_pool, size=N, p=popularity),
    'cuisine_type': np.random.choice(cuisines, size=N, p=cuisine_probs),
    'cost_of_the_order': np.round(np.random.uniform(4.47, 35.41, size=N), 2),
    'day_of_the_week': days,
    'rating': np.random.choice(['5', '4', '3', 'Not given'], size=N, p=[0.39, 0.24, 0.06, 0.31]),
    'food_preparation_time': np.random.randint(20, 36, size=N),
    'delivery_time': np.random.randint(15, 33, size=N),
})

df.to_csv('/mnt/user-data/outputs/foodhub_order_DUMMY.csv', index=False)
print(df.shape)
print(df.head())
print(df.dtypes)
