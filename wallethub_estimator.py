#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:57:02 2019

@author: muralikorrapati
"""

##########################################
import pandas as pd
import numpy as np
import argparse
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

helper1 = None

filter_relevant_features = ['x005',
 'x022',
 'x046',
 'x226',
 'x227',
 'x228',
 'x235',
 'x236',
 'x244',
 'x249',
 'y']

cols_with_missing_vals = ['x002',
 'x003',
 'x004',
 'x005',
 'x044',
 'x045',
 'x234',
 'x235',
 'x272',
 'x287']

class WalletHubEstimator(object):

    def __init__(self, models, params, transformers=None):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.transformers = transformers if transformers is not None else []
        self.transformer_keys = transformers.keys()
        self.grid_searches = {}
        self.score_summary = pd.DataFrame()
        self.best_model_key = None
        
    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=True):
        if not self.score_summary.empty:
            self.score_summary = pd.DataFrame()
            
        transformer_steps = [(k,v) for k, v in self.transformers.items()]
        print("Applying transformers %s" % self.transformer_keys)
        for key in self.keys:
            print("*****Running GridSearchCV for %s*****" % key)
            model = self.models[key]
            params = self.params[key]
            pipeline_steps = transformer_steps.copy()
            pipeline_steps.append(('estimator', model))
            grid_params = {}
            for k, v in params.items():
                grid_params['estimator__{}'.format(k)] = v
            ##
            # Make the pipeline, this allows tuning of parameters of each component via GridsearchCV
            ##
            pipe = Pipeline(steps=pipeline_steps)

            ##
            # Optimize hyper-params, uncomment to tune all of them..
            ##
            gsc = GridSearchCV(
                estimator=pipe,
                param_grid=grid_params,
                cv=cv, 
                n_jobs=n_jobs,
                iid=False,
                verbose=verbose, 
                scoring=scoring, 
                refit=False,
                return_train_score=True
            )
            #gsc.fit(X,y)
            grid_result = gsc.fit(X, y)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            for test_mean, test_stdev, train_mean, train_stdev, param in zip(
                    grid_result.cv_results_['mean_test_score'],
                    grid_result.cv_results_['std_test_score'],
                    grid_result.cv_results_['mean_train_score'],
                    grid_result.cv_results_['std_train_score'],
                    grid_result.cv_results_['params']):
                print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))

            self.grid_searches[key] = gsc    

    def get_score_summary(self, sort_by='mean_score'):
        if not self.score_summary.empty:
            return self.score_summary.sort_values([sort_by], ascending=False)
        
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        self.score_summary = df[columns]
        if self.score_summary.shape[0] > 0:
            self.best_model_key = self.score_summary.iloc[0]['estimator']
            
        return self.score_summary
    
    def best_model(self):
        if self.best_model_key is not None:
            return self.best_model_key, self.grid_searches[self.best_model_key]
        else: # assert
            raise ValueError("Estimators are not fit yet!")
    
    def refit_best_model(self, X, y):
        if self.best_model_key is not None:
            best_grid = self.grid_searches[self.best_model_key]
            pipe = best_grid.estimator
            ##
            # Now train a model with the optimal params we found
            ## 
            pipe.set_params(**best_grid.best_params_)
            pipe.fit(X, y)
        else: # assert
            raise ValueError("GridSearchCV is not performed yet!")
            
    def predict(self, y_test):
        if self.best_model_key is not None:
            best_grid = self.grid_searches[self.best_model_key]
            pipe = best_grid.estimator
            ##
            # Now predict using model with optimal parameters
            ## 
            pred = pipe.predict(y_test)
            return pred
        else: # assert
            raise ValueError("Best model is not refit yet!")
            
def my_accuracy_score(y_true, y_pred, val_range=3):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. Custom loss undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        
    error = np.abs(y_true - y_pred)
    range_check = lambda x: 0 if x>3 else 1
    vfunc = np.vectorize(range_check)
    predict_diffs = vfunc(error)
    return (np.sum(predict_diffs)/predict_diffs.shape[0]) if predict_diffs.shape[0] > 0 else 0
    #return np.sum(predict_score), predict_score.shape[0]


def mean_square_error_range_func(y_true, y_pred, val_range=3):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. Custom loss undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        
    error = np.abs(y_true - y_pred)
    range_check = lambda x: x #if x>3 else 0
    vfunc = np.vectorize(range_check)
    predict_diffs = vfunc(error)
    
    return np.sqrt(np.mean((predict_diffs)**2)) * -1


# Load saved model
def load_model():
    global helper1
    try:
        if helper1 is None:
            # Load the pickle file
            helper1 = joblib.load('wallethub_predictor.pkl') 
    except Exception as e:
        raise ValueError("Failed to load model from disk: %s" % e)


def model_predict(file_path):
    """The entrypoint for the script"""
    load_model()

    # Read file
    print("Loading data...")
    test_df = pd.read_csv(file_path)
    
    # Preprocess data
    print("Preprocessing features...")
    # Fill missing values
    cols_with_missing_vals = [column for column in test_df if test_df[column].count() / len(test_df) < 1]
    
    if 'x287' in filter_relevant_features:
        test_df['x287'].fillna(1.0, axis=0, inplace=True)
        
    for c in filter_relevant_features:
        if c in cols_with_missing_vals:
            if c != 'x287':
                test_df[c].fillna(test_df[c].mean(), inplace=True, axis=0)
            else:
                test_df['x287'].fillna(1.0, axis=0, inplace=True)
            
    #Drop extra columns
    data = test_df[filter_relevant_features]
    #print(data.columns)
    # Predict
    y_test = data['y']
    X_test = data.drop(['y'], axis=1)
    
    global helper1
    t0=time()
    y_pred = np.rint(helper1.predict(X_test))
    t1=time()
    y_test = y_test.values.ravel()
    
    # model evaluation
    rmse = mean_squared_error(y_test, y_pred)
    #accuracy = r2_score(y_test, y_pred)
    accuracy = my_accuracy_score(y_test, y_pred)
    
    # Write preds to file
    print("Predicting targets...")
    file_path_splits = file_path.split('.')
    pred_file_name = file_path_splits[0] + '_pred.' + file_path_splits[1]
    y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])
    y_pred_df.to_csv(pred_file_name)
    print("Saved predictions to file: {}".format(pred_file_name))
    
    # Print RMSE and accuracy
    np.set_printoptions(precision=2)
    print("RMSE: {}".format(rmse))
    np.set_printoptions(precision=2)
    print("Accuracy: {}".format(accuracy*100))
    print("Predict time:", round(t1-t0, 3), "s")


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-path','--path', action="store", help='Path to test samples file', required=True)
    args = vars(parser.parse_args())
    model_predict(args['path'])