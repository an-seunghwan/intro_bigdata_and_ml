#%%
"""
Dataset link: https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data
"""
#%%
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#%%
df = pd.read_csv("./data/personality_datasert.csv")
df
#%%
df['Stage_fear'] = df['Stage_fear'].map({'Yes':1, 'No':0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1, 'No':0})
df['Personality'] = df['Personality'].map({'Extrovert':1, 'Introvert':0})
#%%
X = df.drop('Personality', axis=1) # data point
y = df['Personality'] # label

continuous = [
    'Time_spent_Alone', 'Social_event_attendance',
    'Going_outside', 'Friends_circle_size',
    'Post_frequency']
categorical = [x for x in df.columns if x not in continuous]

# Split dataset into training and test data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
#%%
scaler = MinMaxScaler()
scaler.fit(X_train[continuous])
X_train[continuous] = scaler.transform(X_train[continuous])
X_test[continuous] = scaler.transform(X_test[continuous])
#%%
model = LogisticRegression(random_state=42, fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) # probability
#%%
print(f"Intercept: {model.intercept_[0]:.1f}")
print("Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_[0]):
    print(f"    {feature}: {coef:.1f}")
#%%
"""test accuracy"""
accuracy = (y_pred == y_test).mean().item()
print(f"Accuracy(%): {accuracy*100:.2f}%")
#%%
"""K-fold Cross-validation"""
from sklearn.model_selection import KFold
K = 5
kfold = KFold(n_splits=K, shuffle=True, random_state=42)
#%%
features = X_train.columns.tolist()
cv_accuracy = {}

for feature_to_drop in features:
    print(f"\nEvaluating model without feature: {feature_to_drop}")
    
    # model candidate
    X_cv = X_train.drop(columns=[feature_to_drop]).copy()
    y_cv = y_train.copy()

    fold_accuracies = []
    for train_idx, val_idx in kfold.split(X_cv):
        
        # Build one training set out of folds $1, ..., k-1, k+1, ..., K$
        X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

        # Train model on this training set
        model = LogisticRegression(random_state=42, fit_intercept=True)
        model.fit(X_tr, y_tr)

        # Compute the validation error on fold $k$
        y_val_pred = model.predict(X_val)
        acc = (y_val_pred == y_val).mean().item()
        fold_accuracies.append(acc)

    # Compute the average validation error over the folds
    avg_acc = np.mean(fold_accuracies)
    cv_accuracy[feature_to_drop] = avg_acc
    print(f"    Average CV Accuracy: {avg_acc*100:.2f}%")
#%%
# Select the best model
best_feature_to_drop, best_cv_accuracy = max(cv_accuracy.items(), key=lambda x: x[1])
print(f"Best model drops feature: '{best_feature_to_drop}' with CV Accuracy = {best_cv_accuracy*100:.2f}%")

# Retrain on full training set 
selected_X_train = X_train.drop(columns=[best_feature_to_drop]).copy()
selected_X_test = X_test.drop(columns=[best_feature_to_drop]).copy()

final_model = LogisticRegression(random_state=42, fit_intercept=True)
final_model.fit(selected_X_train, y_train)

# Evaluate on the test set
y_test_pred = final_model.predict(selected_X_test)
test_accuracy = (y_test_pred == y_test).mean().item()

print(f"\nTest Accuracy of the selected model: {test_accuracy*100:.2f}%")
#%%
print(f"Intercept: {final_model.intercept_[0]:.1f}")
print("Coefficients:")
for feature, coef in zip(selected_X_train.columns, final_model.coef_[0]):
    print(f"    {feature}: {coef:.1f}")
#%%