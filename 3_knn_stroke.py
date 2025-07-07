#%%
"""
Dataset link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
"""
#%%
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#%%
df = pd.read_csv("./data/healthcare-dataset-stroke-data.csv")
df = df.drop('id', axis=1)
df = df.dropna()
df
#%%
X = df.drop('stroke', axis=1) # data point
y = df['stroke'] # label

continuous = ['age', 'avg_glucose_level', 'bmi']
categorical = [x for x in X.columns if x not in continuous]

# Split dataset into training and test data (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
#%%
dist = np.zeros((len(X_test), len(X_train)))
for col in continuous:
    scaler = MinMaxScaler()
    train = scaler.fit_transform(X_train[[col]])
    test = scaler.transform(X_test[[col]])
    dist += np.abs(test - train.T)
for col in categorical:
    dist += np.not_equal(
        X_test[col].to_numpy()[:, None], 
        X_train[col].to_numpy()[None, :]
    ).astype(float)

dist /= X.shape[1]
#%%
k = 3
# sorting & nearest neighbors
neighbors = dist.argsort(axis=1)[:, :k]
# the majority label
predict = []
for row in neighbors:
    labels = y_train.iloc[row]
    pred_label = labels.mode().item()
    predict.append(pred_label)
predict = np.array(predict)
#%%
"""test accuracy"""
predict.sum().item()
accuracy = (predict == y_test).mean()
print(f"Accuracy(%, k={k}): {accuracy*100:.2f}%")
#%%
"""other metrics"""
print(classification_report(y_test, predict))
#%%