#%%
"""
Dataset link: https://www.kaggle.com/datasets/osuolaleemmanuel/ad-ab-testing/data
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest
#%%
df = pd.read_csv("./data/AdSmart.csv")
df['experiment'] = df['experiment'].apply(lambda x: 'treatment' if x == 'exposed' else x)
df = df.loc[df['yes'] + df['no'] != 0] # remove no respose users
df
#%%
order = ['control', 'treatment']
counts = df['experiment'].value_counts().reindex(order)
plt.bar(counts.index, counts.values, color=['tab:blue', 'tab:orange'])
plt.xlabel('Group')
plt.ylabel('Count')
plt.title('Number of Participants by Group')
plt.show()
#%%
grouped = df.groupby(['date', 'experiment']).size().unstack()
grouped = grouped.sort_index()

grouped.plot(kind='line', marker='o', figsize=(9, 5))
plt.xlabel('Date')
plt.ylabel('Number of Users')
plt.title('Number of Users per Experiment Group by Date')
plt.xticks(rotation=45)
plt.legend(title='Experiment Group')
plt.tight_layout()
plt.show()
#%%
total_counts = df.groupby('experiment').size()
yes_counts = df.groupby('experiment')['yes'].sum()
yes_rate = yes_counts / total_counts
yes_rate = yes_rate.reindex(['control', 'treatment'])

plt.bar(yes_rate.index, yes_rate.values, color=['tab:blue', 'tab:orange'])
plt.ylim(0, 1)
plt.xlabel('Group')
plt.ylabel('Count')
plt.title('Yes Rate by Group')
plt.show()
#%%
n_A = df.loc[df['experiment'] == 'control'].shape[0]
n_B = df.loc[df['experiment'] == 'treatment'].shape[0]
count_A = df.loc[df['experiment'] == 'control']['yes'].sum().item()
count_B = df.loc[df['experiment'] == 'treatment']['yes'].sum().item()
#%%
"""
H0: p_A = p_B
H1: p_A < p_B
"""
# 1. sample proportion
p_A = count_A / n_A
p_B = count_B / n_B

# 2. pooled proportion
p = (count_A + count_B) / (n_A + n_B)

# 3. standard error
SE = (p * (1 - p) * (1/n_A + 1/n_B)) ** 0.5

# 4. z-statistic
z = (p_A - p_B) / SE

# 5. p-value (one-sided)
p_value = norm.cdf(z).item()

print(f"z-statistic: {z:.4f}")
print(f"p-value: {p_value:.4f}")
#%%
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in conversion rates.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in conversion rates.")
#%%