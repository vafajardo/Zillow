#!/usr/bin/env python

import pandas as pd
from pandas import DataFrame,Series
import numpy as np

# sklearn stuff
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# A custom transformer, which selects certain variables
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, desired_cols):
        self.desired_cols = desired_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.desired_cols].values

# A custom transformer, which first selects the categorical variables
# from the DataFrame and then performs the dummification
class DF_Selector_GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, cat_dict):
        self.cat_dict = cat_dict
        self.ndummies = sum(len(c) - 1  for c in cat_dict.values())
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.fillna(-1) # missing values are given -1 missing label
        foo = np.zeros((X.shape[0],self.ndummies))
        start = 0
        end = 0
        for c in sorted(self.cat_dict.keys()):
            end += len(self.cat_dict[c]) - 1
            foo[:, start:end] = pd.get_dummies(X[c].astype('category', categories=self.cat_dict[c]))[self.cat_dict[c][1:]]
            start += len(self.cat_dict[c]) - 1
        return foo

class Dummify_and_Interact(BaseEstimator, TransformerMixin):
    def __init__(self, interact_pairs, cat_dict):
        self.interact_pairs = interact_pairs
        self.cat_dict = cat_dict
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        features = None
        for pair in self.interact_pairs:
            x1,x2 = pair
            # impute x2 if missing
            imputer = Imputer()
            if np.isnan(X[x2]).any():
                x2vals = imputer.fit_transform(X[[x2]])
            else:
                x2vals = X[[x2]].as_matrix()
            # dummify x1 and multiply by x2vals
            bar = ((pd.get_dummies(X[x1].astype('category',
                    categories=self.cat_dict[x1]))[self.cat_dict[x1][1:]]).as_matrix() * x2vals)
            if features is not None:
                features = np.concatenate((features,bar),axis=1)
            else:
                features = bar
        return features
