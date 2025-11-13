# create_dataset.py
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, norm
import os

os.makedirs('data', exist_ok=True)

cds_stats = {
    'admit_rate': 0.0391,
    'sat_math_25': 740, 'sat_math_75': 800,
    'sat_ebrw_25': 740, 'sat_ebrw_75': 770,
    'act_25': 33, 'act_75': 35,
    'gpa_avg': 3.96,
    'first_gen_pct': 0.12,
    'legacy_pct': 0.08,
    'athlete_pct': 0.02,
    'in_state_pct': 0.15,
    'race_dist': {'Asian': 0.25, 'White': 0.40, 'Hispanic': 0.15, 'Black': 0.08, 'Other': 0.12},
    'gender_dist': {'M': 0.52, 'F': 0.48}
}

np.random.seed(42)
n = 50000
df = pd.DataFrame()

df['age'] = np.random.choice([17, 18, 19], n, p=[0.05, 0.90, 0.05])
df['gender_M'] = np.random.choice([0, 1], n, p=[0.48, 0.52])
df['in_state'] = np.random.binomial(1, 0.15, n)

races = np.random.choice(list(cds_stats['race_dist'].keys()), n, p=list(cds_stats['race_dist'].values()))
for race in cds_stats['race_dist']:
    df[f'race_{race}'] = (races == race).astype(int)

a_math = (740 - 770) / 30; b_math = (800 - 770) / 30
df['sat_math'] = truncnorm(a_math, b_math, loc=770, scale=30).rvs(n).astype(int)

a_ebrw = (740 - 755) / 30; b_ebrw = (770 - 755) / 30
df['sat_ebrw'] = truncnorm(a_ebrw, b_ebrw, loc=755, scale=30).rvs(n).astype(int)

a_act = (33 - 34) / 1.5; b_act = (35 - 34) / 1.5
df['act_composite'] = truncnorm(a_act, b_act, loc=34, scale=1.5).rvs(n).round().astype(int)

df['gpa'] = np.clip(norm(3.96, 0.15).rvs(n), 3.0, 4.0).round(2)
df['extracurriculars_count'] = np.random.poisson(3, n).clip(0, 5)
df['first_gen'] = np.random.binomial(1, 0.12, n)
df['legacy'] = np.random.binomial(1, 0.08, n)
df['athlete'] = np.random.binomial(1, 0.02, n)

base_probs = (
    0.01 +
    0.15 * (df['gpa'] - 3.0) +
    0.10 * (df['sat_math'] - 700) / 100 +
    0.08 * (df['sat_ebrw'] - 700) / 100 +
    0.05 * df['extracurriculars_count'] / 5 +
    0.03 * df['first_gen'] +
    0.20 * df['legacy'] +
    0.60 * df['athlete']
)
probs = np.clip(base_probs, 0, 0.99)
df['admit'] = np.random.binomial(1, probs)

print(f"Admit rate: {df['admit'].mean():.4f}")
df.to_csv('data/stanford_admissions.csv', index=False)
print("Saved!")