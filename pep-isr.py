import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

dataset = pd.read_csv("/Users/alert/Downloads/C-PEPTIDE/diabetes_data 2.csv")
dataset_label = dataset["Insulin_Levels"]
dataset.drop(columns=["Insulin_Levels","Unnamed: 0"], inplace=True)
dataset_new = pd.get_dummies(dataset, columns=['Family_History_of_Diabetes', 'Gestational_Diabetes',
'Physical_Activity', 'Smoking', 'Alcohol_Consumption',
       'Obesity', 'Diet', 'Sleep_Apnea', 'Diabetes_Status', 'Hypertension', 'PCOS'])


X = dataset_new
y = dataset_label

model = Lasso(alpha=0.1)
model.fit(X, y)
selected_features = X.columns[model.coef_ != 0]
X_new = X[selected_features]


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)


preprocessing = ColumnTransformer(transformers=[
    ("scale", StandardScaler(), selected_features)
])

base_models = [
    ("rdf", RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=2, max_depth=5)),
    ("kn", KNeighborsRegressor(n_neighbors=5, weights="distance")),
]


meta_model = LinearRegression()

stacked_model = StackingRegressor(estimators=base_models,
                                   final_estimator=meta_model, passthrough=True)

xgbmodel = Pipeline(steps=[
    ("preprocess", preprocessing),
    ("model", stacked_model)
])

xgbmodel.fit(X_train, y_train)
prediction = xgbmodel.predict(X_test)

print(root_mean_squared_error(y_test, prediction))
print(mean_absolute_percentage_error(y_test, prediction))


target = y_test  # replace 'target' with your column name

min_val = target.min()
max_val = target.max()
range_val = max_val - min_val

print("Minimum:", min_val)
print("Maximum:", max_val)
print("Range:", range_val)

# Optional: calculate RMSE as percentage of range
rmse = root_mean_squared_error(y_test, prediction)
rmse_percent = (rmse / range_val) * 100
print(f"RMSE is {rmse_percent:.2f}% of the target range")
