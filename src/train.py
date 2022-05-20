"""
Train model.
"""

from copy import copy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.model_selection import train_test_split

from eval import calc_metrics, make_predictions
from sample import sample_negative_examples

 
def training_with_resampling(train, test, df_trans, df_click, catboost_params,
                             sample_size, validation=False, resample_freq=1000, 
                             random_state=42, thread_count=20, plot=False, verbose=False):

    catboost_params  = copy(catboost_params)
    num_iterations = int(np.ceil(catboost_params['iterations']/resample_freq)) - 1
    catboost_params['iterations'] = resample_freq

    if validation:
        train, val = train_test_split(train, test_size=0.1, random_state=42)

        val = sample_negative_examples(
            val, sample_size=sample_size, random_state=random_state)
        val = pd.merge(val, df_trans, left_on='bank', right_index=True)
        val = pd.merge(val, df_click, left_on='rtk', right_index=True)
        val = val.sort_values('bank')

        yval = val.target.values
        queries_val = val.bank.values
        Xval = val.drop(['bank', 'rtk', 'target'], axis=1).values

    train2 = train.copy()
    train2 = sample_negative_examples(
        train2, sample_size=sample_size, random_state=random_state)
    train2 = pd.merge(train2, df_trans, left_on='bank', right_index=True)
    train2 = pd.merge(train2, df_click, left_on='rtk', right_index=True)
    train2 = train2.sort_values('bank')

    ytrain = train2.target.values
    queries_train = train2.bank.values
    Xtrain = train2.drop(['bank', 'rtk', 'target'], axis=1).values

    train_pool = Pool(data=Xtrain, label=ytrain, group_id=queries_train)
    if validation:
        validation_pool = Pool(data=Xval, label=yval, group_id=queries_val)
        print('train shape', Xtrain.shape, 'validation shape', Xval.shape)
    else:
        print('train shape', Xtrain.shape)

    clf = CatBoostRanker(thread_count=thread_count, **catboost_params)
    if validation:
        clf.fit(train_pool, eval_set=validation_pool, plot=plot, verbose=verbose)
    else:
        clf.fit(train_pool, plot=plot, verbose=verbose)

    for i in range(num_iterations):

        print(f'Iteration {i+1}')
        train2 = train.copy()
        train2 = sample_negative_examples(
            train2, sample_size=sample_size, random_state=random_state + 1 + i)
        train2 = pd.merge(train2, df_trans, left_on='bank', right_index=True)
        train2 = pd.merge(train2, df_click, left_on='rtk', right_index=True)
        train2 = train2.sort_values('bank')

        ytrain = train2.target.values
        queries_train = train2.bank.values
        Xtrain = train2.drop(['bank', 'rtk', 'target'], axis=1).values

        train_pool = Pool(data=Xtrain, label=ytrain, group_id=queries_train)
        if validation:
            print('train shape', Xtrain.shape, 'validation shape', Xval.shape)
        else:
            print('train shape', Xtrain.shape)
        
        if validation:
            clf.fit(train_pool, eval_set=validation_pool, plot=plot,
                    verbose=verbose, init_model=clf)
        else:
            clf.fit(train_pool, plot=plot, verbose=verbose, init_model=clf)
    
    if test is None:
        return clf
    else:
        preds = make_predictions(clf, test, df_trans, df_click)
        r1, mrr, precision = calc_metrics(preds, test)
        metrics = {'r1': r1, 'mrr': mrr, 'precision': precision,
                   'best_iteration': clf.best_iteration_}

        return clf, metrics, preds
