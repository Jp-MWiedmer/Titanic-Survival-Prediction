import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import error_analysis_
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, get_scorer_names, precision_recall_curve, confusion_matrix

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', None)
titanic = pd.read_csv('train.csv')


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family', 'Abandoned']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Family'] = - X['SibSp'] + X['Parch']
        X['Abandoned'] = X['Age']*X['Pclass']
        return X[self.features]


class Inputer(BaseEstimator, TransformerMixin):
    def __init__(self, feature, strategy='drop'):
        self.feature = feature
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.strategy == 'drop':
            X.dropna(subset=[self.feature])
        elif self.strategy == 'median':
            median = X[self.feature].median()
            X[self.feature].fillna(median, inplace=True)
        elif self.strategy == 'mean':
            mean = X[self.feature].mean()
            X[self.feature].fillna(mean, inplace=True)
        return X


def extract_deck(cabin):
    if str(cabin)[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return str(cabin)[0]
    else:
        return 'U'


class Cabin_2_Deck(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Deck'] = X['Cabin'].apply(extract_deck)
        return X[['Deck']]


fi = np.array([0.5450425577171888, 0.344524240659434, 0.26266015745728477, 0.18254100478711047,
               0.15750349538578834, 0.13416180414281956, 0.23582317017816445, 1.3607287233183707,
               1.3600144066142867, 0.0026099543128950873, 0.035643010549917434, 0.4566605134776037,
               0.4314843684504174, 0.7143139406855987, 0.4983756327263118, 0.6792208493207268, 0.5406113185969357])


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


def grid_search_SVC(X, y):
    param_grid = [{'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [0.1, 0.5, 1, 5, 10, 100],
                   'decision_function_shape': ['ovo', 'ovr'], 'gamma': ['auto', 'scale']},
                  {'kernel': ['poly'],'degree': [3, 4, 5, 6], 'C': [0.1, 0.5, 1, 5, 10, 100],
                   'decision_function_shape': ['ovo'], 'gamma': ['auto', 'scale']}
                  ]
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose=3)
    grid_search.fit(X, y)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)


def grid_search_k(X, y):
    param_grid = [{'selection__k': list(range(1, 18))}]

    grid_search = GridSearchCV(remaining_pipeline, param_grid, cv=5,
                                    scoring='accuracy')
    grid_search.fit(X, y)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

# PIPELINES

num_prepro = Pipeline([('inputer', Inputer(feature='Age', strategy='median')),
                       ('inputer2', Inputer(feature='Fare', strategy='median')),
                       ('adder', FeatureAdder()),
                       ('scaler', StandardScaler())])

cabin_prepro = Pipeline([('deck', Cabin_2_Deck()),
                         ("encoder", OneHotEncoder())])

data_prepro = ColumnTransformer([
    ("num", num_prepro, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ("cat1", OneHotEncoder(), ['Sex']),
    ("cat2", cabin_prepro, ['Cabin'])
])

remaining_pipeline = Pipeline([
    ('selection', TopFeatureSelector(fi, k=17)),
    ('prediction', SVC(decision_function_shape='ovo'))
])

full_pipe = Pipeline([('preprocessing', data_prepro), ('classification', remaining_pipeline)])

y = titanic['Survived']
full_pipe.fit(titanic, y)
y_pred = full_pipe.predict(titanic)

error_analysis_.metrics(y, y_pred)
error_analysis_.plot_confusion_matrix(y, y_pred)


# ANALISE PRECISAO VS RECALL

svm_clf = SVC(decision_function_shape='ovo')
X = data_prepro.transform(titanic)
svm_clf.fit(X, y)
y_pred_df = cross_val_predict(svm_clf, X, y, cv=5, method="decision_function")
error_analysis_.plot_precision_recall(y, y_pred_df, versus=True)
error_analysis_.plot_roc_curve(y, y_pred_df)

y_pred_new = error_analysis_.new_treshold_metrics(y, y_pred_df, metric='precision', value=0.76)
error_analysis_.plot_confusion_matrix(y, y_pred_new)

exit()
# TESTE

titanic_test = pd.read_csv('test.csv')

y_pred = full_pipe.predict(titanic)

with open('output.csv', 'w') as out_file:
    out_file.write('PassengerId,Survived\n')
    i = 892
    for row in y_pred:
        out_file.write(f'{i},{str(row)}\n')
        i += 1


def model_selection(X, y):

    model = LogisticRegression()
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    print(f'LogReg: {accuracy.mean()}: {accuracy}')
    model.fit(X, y)
    importance = [abs(i) for i in model.coef_[0]]
    print(importance)

    model = KNeighborsClassifier()
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    print(f'KNN: {accuracy.mean()}: {accuracy}')

    model = DecisionTreeClassifier(random_state=42)
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    """model.fit(X, y)
    importance = model.feature_importances_
    cat = data_prepro.named_transformers_["cat1"]
    attributes = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family', 'Abandoned'] + list(cat.categories_[0])
    for i in sorted(zip(importance, attributes), reverse=True):
        print(i)"""
    print(f'Tree: {accuracy.mean()}: {accuracy}')

    model = RandomForestClassifier(random_state=42)
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    print(f'RandomForest: {accuracy.mean()}: {accuracy}')

    model = SVC()
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    print(f'SVC: {accuracy.mean()}: {accuracy}')











