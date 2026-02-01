import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

dataset = pd.read_csv("/Users/alert/Downloads/C-PEPTIDE/diabetes_data 2.csv")
dataset_label = dataset["Insulin_Levels"]
dataset.drop(columns=["Insulin_Levels","Unnamed: 0"], inplace=True)
dataset_new = pd.get_dummies(dataset, columns=['Family_History_of_Diabetes', 'Gestational_Diabetes',
'Physical_Activity', 'Smoking', 'Alcohol_Consumption',
       'Obesity', 'Diet', 'Sleep_Apnea', 'Diabetes_Status', 'Hypertension', 'PCOS'])

selected_features = [
    "Diabetes_Status_Negative",
    "Family_History_of_Diabetes_No",
    "LDL_Cholesterol",
    "Proinsulin_Levels",
    "OGTT",
    "Microalbuminuria",
    "AST",
    "Random_Blood_Glucose",
    "CRP_Levels",
    "Fasting_Blood_Glucose",
    "Triglyceride_Levels",
    "Blood_Pressure_Diastolic",
    "Creatinine_Levels",
    "Fructosamine_Levels",
    "Blood_Pressure_Systolic",
    "Hypertension_No",
    "Postprandial_Blood_Glucose",
    "ALT",
    "eGFR",
    "Uric_Acid_Levels",
    "Gestational_Diabetes_No",
    "Physical_Activity_No",
    "Obesity_No",
    "PCOS_No",
    "BMI",
    "Diet_No",
    "Waist_Circumference",
    "HOMA_IR",
    "HbA1c",
    "Smoking_No"
]


def feature_engineering(dataset, label):
    X = dataset
    y = label

    X_new = X[selected_features]
    
    return X_new, y


def train_model(new_dataset, label):
    preprocessing = ColumnTransformer(transformers=[
        ("scale", StandardScaler(), selected_features)
    ])
    X_train, X_test, y_train, y_test = train_test_split(new_dataset, label, test_size=0.2, random_state=42)
    print(X_train)
    base_models = [
        ("rdf", RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=3, max_depth=5, random_state=42)),
        ("svr", SVR(kernel='rbf', C=1, gamma="scale")),
        ("lin", LinearRegression()),
        ("knn", KNeighborsRegressor(n_neighbors=5, weights="uniform"))
    ]

    meta_model = RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=3, max_depth=5, random_state=42)
    stacked_model = StackingRegressor(estimators=base_models,
                                    final_estimator=meta_model, passthrough=True)

    regression_model = Pipeline(steps=[
        ("preprocess", preprocessing),
        ("model", stacked_model)
    ])
    regression_model.fit(X_train, y_train)
    prediction = regression_model.predict(X_test)

    return y_test, prediction

data = feature_engineering(dataset_new, dataset_label)
model_prediction = train_model(data[0], data[1])

print(f'RMSE - {root_mean_squared_error(model_prediction[0], model_prediction[1])}')
print(f'MSE - {mean_squared_error(model_prediction[0], model_prediction[1])}')
