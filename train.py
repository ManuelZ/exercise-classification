# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Local imports
from utils import gen_parameters_from_log_space

search = True

###############################################################################
# Data preparation
###############################################################################
df_train = pd.read_csv("train.csv", index_col="timestamp")
df_test = pd.read_csv("test.csv", index_col="timestamp")
X_train  = df_train.drop(columns=['classe']).copy()
y_train  = df_train.loc[:, 'classe'].copy()
X_test  = df_test.drop(columns=['classe']).copy()
y_test  = df_test.loc[:, 'classe'].copy()

numeric_columns = make_column_selector(dtype_include=np.number)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

###############################################################################
# Training preparation
###############################################################################

scorer = make_scorer(
    score_func = f1_score,
    average    = "macro" 
)

numeric_columns = make_column_selector(dtype_include=np.number)

pipeline = Pipeline([
    ('union', ColumnTransformer(
        [
            (
                'imputer', 
                SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True),
                numeric_columns
            ),
            
            # (
            #     'scaler',
            #     StandardScaler(), 
            #     numeric_columns
            # ),

        ], remainder='drop'
    )),

    ('classifier', RandomForestClassifier()),
], verbose=True)




###############################################################################
# Training
###############################################################################
if search:

    lin_space = np.arange(120, 180, 20, dtype=np.int)

    log_space = gen_parameters_from_log_space(
            low_value  = 0.001,
            high_value = 1,
            n_samples  = 10
        )

    grid = {
        'classifier__n_estimators': lin_space
        # 'classifier' : [
        #     MLPClassifier(),
        #     RandomForestClassifier(),
        #     ExtraTreeClassifier(),
        #     SGDClassifier(),
        # ]
    }

    searcher = GridSearchCV(
        estimator          = pipeline, 
        param_grid         = grid,
        n_jobs             = 7, 
        return_train_score = True, 
        refit              = True,
        verbose            = True,
        cv                 = StratifiedKFold(n_splits=3),
        scoring            = scorer,
    )
    
    model = searcher.fit(X_train, y_train)
    print(f"Best params found: ", searcher.best_params_)

else:
    model = pipeline.fit(X_train, y_train)

print("Finished")

###############################################################################
# Scoring
###############################################################################
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
    normalize      = None,
    ax             = ax
)
plt.show()
