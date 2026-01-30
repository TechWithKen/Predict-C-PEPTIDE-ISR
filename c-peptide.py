import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



dataset = pd.read_csv("./diabetes_data 2.csv")
dataset['Diabetes_Status'] = LabelEncoder().fit_transform(dataset['Diabetes_Status'])
categorical_data = ['Family_History_of_Diabetes', 'Gestational_Diabetes',
'Physical_Activity', 'Smoking', 'Alcohol_Consumption',
       'Obesity', 'Diet', 'Sleep_Apnea']


numeric_data = [
       'HDL_Cholesterol', 'Fructosamine_Levels', 'C_Peptide', 'Proinsulin_Levels', "Insulin_Levels", "HbA1c"]


preprocessing = ColumnTransformer(transformers=[
    ("cat", Pipeline(steps=[
        ("encode", OneHotEncoder(drop="first", sparse_output=False))]), categorical_data),
    ("scale", Pipeline(steps=[
        ('scale', StandardScaler())]), numeric_data)
])
processing = Pipeline(steps=[
    ("preprocess", preprocessing),
    ("model", RandomForestClassifier(
        max_depth=5,
        n_estimators=200,
        max_features="sqrt",
    )),
])
last_col = dataset.columns[-1]
X = dataset[dataset.columns.difference([last_col])]
y = dataset[last_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

processing.fit(X_train, y_train)
prediction = processing.predict(X_test)
print(f' Accuracy: {accuracy_score(y_test, prediction)}, Precision {precision_score(y_test, prediction)}, Recall {recall_score(y_test, prediction)} F1-Score = {f1_score(y_test, prediction)}')

cat_features = preprocessing.named_transformers_['cat'].named_steps['encode'].get_feature_names_out(categorical_data)


all_features = np.concatenate([numeric_data, cat_features])


importances = processing.named_steps['model'].feature_importances_


feature_importances = pd.DataFrame({
    'feature': all_features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

feature_importances.head(15)
confusion_matrix(y_test, prediction)
