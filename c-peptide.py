import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_curve


dataset = pd.read_csv("/Users/alert/Downloads/C-PEPTIDE/diabetes_data 2.csv")
dataset['Diabetes_Status'] = LabelEncoder().fit_transform(dataset['Diabetes_Status'])

last_col = dataset.columns[-1]
X = dataset[dataset.columns.difference([last_col])]
y = dataset[last_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_numeric = X_train.select_dtypes(include=np.number)

kbd = KBinsDiscretizer(
    n_bins=5,
    encode='ordinal',
    strategy='quantile'
)

X_train_binned = kbd.fit_transform(X_train_numeric)
selector = SelectKBest(score_func=chi2, k=15)
selector.fit(X_train_binned, y_train)

selected_features = X_train_numeric.columns[selector.get_support()]
selected_features

categorical_data = ['Family_History_of_Diabetes', 'Gestational_Diabetes',
'Physical_Activity', 'Smoking', 'Alcohol_Consumption',
       'Obesity', 'Diet', 'Sleep_Apnea']


numeric_data = selected_features


preprocessing = ColumnTransformer(transformers=[
    ("cat", Pipeline(steps=[
        ("encode", OneHotEncoder(drop="first", sparse_output=False))]), categorical_data),
    ("scale", Pipeline(steps=[
        ('scale', StandardScaler())]), numeric_data)
])

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
    ("preprocess", preprocessing),
    ("model", stack_models)
])

numeric_selected = X_train[selected_features]
categorical = X_train.select_dtypes(exclude=np.number)

X_train_final = pd.concat([numeric_selected, categorical], axis=1)

processing.fit(X_train_final, y_train)
prediction = processing.predict(X_test)
print(f' Accuracy: {accuracy_score(y_test, prediction)}, Precision {precision_score(y_test, prediction)}, Recall {recall_score(y_test, prediction)} F1-Score = {f1_score(y_test, prediction, average="weighted")}')



confusion_matrix(y_test, prediction)
y_prob = processing.predict_proba(X_test)[:,1] # probability of positive class
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1],[0,1],'--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# AUC score
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)
print("AUC:", auc)
information_gain = mutual_info_classif(numeric_selected, y_train)
information_gain, selected_features