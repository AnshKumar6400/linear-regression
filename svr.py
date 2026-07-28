import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df=pd.read_csv(Path.cwd()/'datasets'/'Housing.csv') 
print(df.head())
scaler=StandardScaler()
binary_cols = ['mainroad', 'guestroom', 'basement', 'prefarea', 'hotwaterheating', 'airconditioning']
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].replace(
    {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
)
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
        'mainroad', 'guestroom', 'basement', 'prefarea',
        'hotwaterheating', 'airconditioning','furnishingstatus']]
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled =scaler.fit_transform(X_train)
param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1]
}

grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_train_scaled, y_train)
print(grid.best_params_)
model = grid.best_estimator_
X_test_scaled = scaler.transform(X_test)
y_pred=model.predict(X_test_scaled)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))