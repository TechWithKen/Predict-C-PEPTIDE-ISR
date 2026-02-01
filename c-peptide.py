import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_curve

# Encode the label class.
dataset = pd.read_csv("/Users/alert/Downloads/C-PEPTIDE/diabetes_data 2.csv")
dataset['Diabetes_Status'] = LabelEncoder().fit_transform(dataset['Diabetes_Status'])


def discreteSplit(dataframe):
    last_col = dataframe.columns[-1]
    X = dataframe[dataframe.columns.difference([last_col])]
    y = dataframe[last_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_numeric = X_train.select_dtypes(include=np.number)
    kbd = KBinsDiscretizer(
        n_bins=4,
        encode='ordinal',
        strategy='quantile'
    )
    X_train_binned = kbd.fit_transform(X_train_numeric)
    selector = SelectKBest(score_func=chi2, k=20)
    selector.fit(X_train_binned, y_train)
    selected_features = X_train_numeric.columns[selector.get_support()]

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "selected_features": selected_features}



def feature_engineering():
    categorical_data = ['Family_History_of_Diabetes', 'Gestational_Diabetes',
    'Physical_Activity', 'Smoking', 'Alcohol_Consumption',
        'Obesity', 'Diet', 'Sleep_Apnea']
    numeric_data = discreteSplit(dataset)["selected_features"]


    preprocessing = ColumnTransformer(transformers=[
        ("cat", Pipeline(steps=[
            ("encode", OneHotEncoder(drop="first", sparse_output=False))]), categorical_data),
        ("scale", Pipeline(steps=[
            ('scale', StandardScaler())]), numeric_data)
    ])

    return preprocessing


def train_ensemble_models(dictionary):
    base_models = [
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ("rfb", RandomForestClassifier(n_estimators=200, criterion="entropy", max_features="sqrt", max_depth=10, random_state=42)),
        ("dct", DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=10, max_features="sqrt", random_state=42)),
        ("xgb", XGBClassifier(learning_rate=0.8, n_estimators=200, max_depth=10, random_state=42)),
        ("svc", SVC(C=1, kernel="rbf", probability=True, random_state=42, gamma="scale")),
    ]

    meta_model = LogisticRegression(C=10, class_weight="balanced")

    stack_models = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=True
    )
    processing = Pipeline(steps=[
        ("preprocess", feature_engineering()),
        ("model", stack_models)
    ])

    numeric_selected = dictionary["X_train"][dictionary["selected_features"]]
    categorical = dictionary["X_train"].select_dtypes(exclude=np.number)

    X_train_final = pd.concat([numeric_selected, categorical], axis=1)

    processing.fit(X_train_final, dictionary["y_train"])
    prediction = processing.predict(dictionary["X_test"])

    model_dataset = discreteSplit(dataset)
    measurement = model_dataset["y_test"], prediction
    print(f' Accuracy: {accuracy_score(measurement[0], measurement[1])}, Precision {precision_score(measurement[0], measurement[1])}, Recall {recall_score(measurement[0], measurement[1])}, F1-Score = {f1_score(measurement[0], measurement[1], average="weighted")}')
    y_prob = processing.predict_proba(model_dataset["X_test"])[:,1] # probability of positive class
    auc = roc_auc_score(model_dataset["y_test"], y_prob)
    return("AUC:", auc)

print(train_ensemble_models(discreteSplit(dataset)))


"""
METRICS ANALYSIS:

Accuracy: 92.6%
Precision: 92.2% 
Recall: 94.5%
F1-Score = 92.6%
Area Under Curve = 98.4%

"""