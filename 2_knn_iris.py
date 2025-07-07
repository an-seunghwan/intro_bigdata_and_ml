#%%
"""
Dataset link: https://archive.ics.uci.edu/dataset/53/iris
"""
#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#%%
"""load data"""
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("./data/iris/iris.data", header=None)
df.columns = column_names
df
#%%
"""pre-process"""
df['class'] = df['class'].map(
    {x:i for i,x in enumerate(df['class'].unique())}
)
df
X = df.drop('class', axis=1) # data point
y = df['class'] # label

# Split dataset into training and test data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
#%%
for k in [5, 51]:
    """predict on test data points"""
    predict = []
    for i in tqdm(range(len(X_test))):
        # distance function: Euclidean distance
        dist = np.sqrt((X_train - X_test.iloc[i]).pow(2)).sum(axis=1)
        # sorting & nearest neighbors
        neighbors = dist.argsort()[:k]
        # the majority label
        pred = y_train.iloc[neighbors].mode().item()
        predict.append(pred)
    
    """test accuracy"""
    accuracy = (predict == y_test).mean()
    print(f"Accuracy(%, k={k}): {accuracy*100:.2f}%")
    print()
#%%