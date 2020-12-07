import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from catboost import CatBoostClassifier
from catboost import Pool
from catboost import cv


df = pd.read_csv("processed.csv", index_col="timestamp")
df = df.drop(columns=['user_name']).copy()
X  = df.drop(columns=['classe']).copy()
y  = df.loc[:, 'classe'].copy()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.33, random_state=42)
)

search = False

cv_params = {
    'iterations': 100, # max number of trees
    'learning_rate': 1,
    'random_seed': 42,
    'logging_level': 'Verbose',
    'loss_function': 'MultiClass',
    'task_type': 'CPU',              
}

print("Training")
model = CatBoostClassifier(**cv_params)


if search:

    search_grid = {
        'iterations' : [100, 200, 300]
    }

    grid_search_result = model.grid_search(
        param_grid = search_grid,
        X          = X_train.values,
        y          = y_train.values.ravel(),
        verbose    = True
    )

else:
    train_pool = Pool(X_train, y_train)
    model.fit(train_pool, plot=False)
    y_pred = model.predict(X_test)

    scorer = make_scorer(
        score_func = f1_score,
        average    = "macro" 
    )

    train_score = scorer(model, X_train, y_train)
    print(f"Train score: {train_score:.2f}")

    test_score = scorer(model, X_test, y_test)
    print(f"Test score: {test_score:.2f}")

    fig,ax = plt.subplots(1,1, figsize=(4,4))
    labels = le.inverse_transform(model.classes_)

    plot_confusion_matrix(
        estimator      = model,
        X              = X_test,
        y_true         = y_test,
        display_labels = labels,
        cmap           = plt.cm.Blues,
        normalize      = 'true',
        ax             = ax
    )
    plt.show()


print("Finished")

