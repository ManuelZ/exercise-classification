import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from catboost import CatBoostClassifier
from catboost import Pool
from catboost import cv


###############################################################################
# Data load
###############################################################################

df_train = pd.read_csv("train.csv", index_col="timestamp")
X_train  = df_train.drop(columns=['classe']).copy()
y_train  = df_train.loc[:, 'classe'].copy()

df_test = pd.read_csv("test.csv", index_col="timestamp")
X_test  = df_test.drop(columns=['classe']).copy()
y_test  = df_test.loc[:, 'classe'].copy()

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



###############################################################################
# Training / Searching of hyperparameters
###############################################################################

search = False

train_params = {
    'iterations'    : 50, # max number of trees
    'learning_rate' : 0.3,
    'depth'         : 10,
    'l2_leaf_reg'   : 1,
    'random_seed'   : 42,
    'logging_level' : 'Verbose',
    'loss_function' : 'MultiClass',
    'task_type'     : 'GPU',              
}

print("Training")
model = CatBoostClassifier(**train_params)


if search:

    search_grid = {
        'iterations'    : [20, 50],
        'learning_rate' : [0.2, 0.3, 0.4],
        'depth'         : [4, 6, 10],
        'l2_leaf_reg'   : [0.5, 1]
    }

    grid_search_result = model.grid_search(
        param_grid = search_grid,
        cv         = StratifiedKFold(n_splits=3),
        X          = X_train,
        y          = y_train,
        refit      = True,
        stratified = True,
        verbose    = True
    )
    print(f"Best params:\n{grid_search_result['params']}")
    print(pd.DataFrame(grid_search_result['cv_results']))

else:
    train_pool = Pool(X_train, y_train)
    model.fit(train_pool, plot=False)


###############################################################################
# Scoring on test set
###############################################################################
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

