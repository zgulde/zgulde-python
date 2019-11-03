from zgulde.ds_imports import *

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# X, y = iris.drop(columns='species'), iris.species.astype('category').cat.codes

X, y = mpg[['hwy', 'cty', 'displ']], mpg.cyl

X_train, X_test, y_train, y_test = train_test_split(X, y)

models = [(DecisionTreeClassifier, {'max_depth': [2, 3, 4],
                                    'max_features': [None, 1, 3]}),
          (KNeighborsClassifier, {'n_neighbors': [1, 3, 5, 7]}),
          (LogisticRegression, {'C': [.01, .1, 1, 10, 100, 1000],
              'solver': ['lbfgs']})]

# results = []
# for model, params in models:
#     grid = GridSearchCV(model(), params, cv=5).fit(X, y)
#     params, scores = grid.cv_results_['params'], grid.cv_results_['mean_test_score']
#     for p, s in zip(params, scores):
#         p['score'] = s
#         p['model'] = model.__name__
#     results.extend(params)

results = []
for model, params in models:
    grid = GridSearchCV(model(), params, cv=5).fit(X, y)
    params, scores = grid.cv_results_['params'], grid.cv_results_['mean_test_score']
    for p, s in zip(params, scores):
        results.extend([{
            'score': s, 'model': model.__name__, 'params': p
        }])

pd.DataFrame(results).sort_values(by='score')

partial(LogisticRegression, solver='lbfgs', multi_class='auto')
