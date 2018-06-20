# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

regressor_config_dict = {

# Not wrapped
#    'sklearn.linear_model.ElasticNetCV': {
#        'l1_ratio': np.arange(0.0, 1.01, 0.05),
#        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#    },

    'd3m.primitives.sklearn_wrap.SKExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'd3m.primitives.sklearn_wrap.SKGradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'd3m.primitives.sklearn_wrap.SKAdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"],
        'max_depth': range(1, 11)
    },

    'd3m.primitives.sklearn_wrap.SKDecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'd3m.primitives.sklearn_wrap.SKKNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

# Not wrapped
#    'sklearn.linear_model.LassoLarsCV': {
#        'normalize': [True, False]
#    },

    'd3m.primitives.sklearn_wrap.SKLinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'd3m.primitives.sklearn_wrap.SKRandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

# Not wrapped
#    'sklearn.linear_model.RidgeCV': {
#    },

# Not wrapped
#    'xgboost.XGBRegressor': {
#        'n_estimators': [100],
#        'max_depth': range(1, 11),
#        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
#        'subsample': np.arange(0.05, 1.01, 0.05),
#        'min_child_weight': range(1, 21),
#        'nthread': [1]
#    },

    # Preprocesssors
# Not wrapped
#    'sklearn.preprocessing.Binarizer': {
#        'threshold': np.arange(0.0, 1.01, 0.05)
#    },

    'd3m.primitives.sklearn_wrap.SKFastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'd3m.primitives.sklearn_wrap.SKFeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    },

# Not wrapped
#    'sklearn.preprocessing.MaxAbsScaler': {
#    },

    'd3m.primitives.sklearn_wrap.SKMinMaxScaler': {
    },

# Not wrapped
#    'sklearn.preprocessing.Normalizer': {
#        'norm': ['l1', 'l2', 'max']
#    },

    'd3m.primitives.sklearn_wrap.SKNystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'd3m.primitives.sklearn_wrap.SKPCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'd3m.primitives.sklearn_wrap.SKPolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'd3m.primitives.sklearn_wrap.SKRBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

# Not wrapped
#    'sklearn.preprocessing.RobustScaler': {
#    },

    'd3m.primitives.sklearn_wrap.SKStandardScaler': {
    },

    'd3m.primitives.sri.tpot.ZeroCount': {
    },

    'd3m.primitives.sklearn_wrap.SKOneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False]
    },

    # Selectors
# Not wrapped
#    'sklearn.feature_selection.SelectFwe': {
#        'alpha': np.arange(0, 0.05, 0.001),
#        'score_func': {
#            'sklearn.feature_selection.f_regression': None
#        }
#    },

    'd3m.primitives.sklearn_wrap.SKSelectPercentile': {
        'percentile': range(1, 100),
        'score_func': ['f_regression']
    }

# Not wrapped
#    'sklearn.feature_selection.VarianceThreshold': {
#        'threshold': np.arange(0.05, 1.01, 0.05)
#    },

# Not wrapped
#    'sklearn.feature_selection.SelectFromModel': {
#        'threshold': np.arange(0, 1.01, 0.05),
#        'estimator': {
#            'sklearn.ensemble.ExtraTreesRegressor': {
#                'n_estimators': [100],
#                'max_features': np.arange(0.05, 1.01, 0.05)
#            }
#        }
#    }

}
