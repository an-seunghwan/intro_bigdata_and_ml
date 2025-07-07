#%%
"""
Dataset link: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
"""
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv("./data/Student_Performance.csv")
df.isna().sum()
#%%
"""pre-processing"""
encoder = LabelEncoder()
df["Extracurricular Activities"] =  encoder.fit_transform(df["Extracurricular Activities"])
#%%
X = df.drop('Performance Index', axis=1) # input
y = df['Performance Index'] # output

plt.hist(y)
plt.show()
plt.close()

# Split dataset into training and test data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
#%%
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#%%
print(f"Intercept: {model.intercept_:.1f}")
print("Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"    {feature}: {coef:.1f}")
#%%
# """Test MSE"""
# # from sklearn.metrics import mean_squared_error
# # mse = mean_squared_error(y_test, y_pred)
# mse = ((y_test - y_pred) ** 2).mean().item()
# print(f"Mean Squared Error on Test Set: {mse:.1f}")
#%%
continuous = [
    'Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'
]
discrete = ['Extracurricular Activities']

plt.figure(figsize=(8, 5))
sns.boxplot(data=X_train[continuous])
plt.xticks(rotation=45)
plt.title("Variable Scale Range")
plt.tight_layout()
plt.show()
plt.close()
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[continuous])
X_train[continuous] = scaler.transform(X_train[continuous])
X_test[continuous] = scaler.transform(X_test[continuous])
#%%
plt.figure(figsize=(8, 5))
sns.boxplot(data=X_train[continuous])
plt.xticks(rotation=45)
plt.title("Variable Scale Range (after scaling)")
plt.tight_layout()
plt.show()
plt.close()
#%%
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#%%
print(f"Intercept: {model.intercept_:.1f}")
print("Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"    {feature}: {coef:.1f}")
#%%
"""Test MSE"""
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, y_pred)
mse = ((y_test - y_pred) ** 2).mean().item()
print(f"Mean Squared Error on Test Set: {mse:.1f}")
#%%
"""Test MAPE"""
mape = ((y_test - y_pred) / y_test).abs().mean().item() * 100
print(f"Mean Absolute Percentage Error on Test Set: {mape:.1f}%")
#%%