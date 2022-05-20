import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker

from aggregate import aggregate_clickstream, aggregate_transactions
from preprocess import click_preprocess, filter_features, trans_preprocess


WEIGHTS = None
RANK = [True, True, True, True, True]
CLICK_DATA = ['grouped_week', 'grouped_date', 'grouped_week', 'grouped_week', 'weekly_normed']
TRANS_DATA = ['grouped', 'grouped_date', 'grouped_week', 'grouped_week', 'weekly_normed']
CLICK_PARAMS = [{'cat_id': True, 'level_0': False, 'level_1': False,
                 'level_2': False, 'normed': True},
                {'cat_id': True, 'level_0': False, 'level_1': False,
                 'level_2': False, 'normed': True},
                {'cat_id': False, 'level_0': True, 'level_1': True,
                 'level_2': True, 'normed': True},
                {'cat_id': True, 'level_0': False, 'level_1': False,
                 'level_2': False, 'normed': True},
                {'cat_id': True, 'level_0': False, 'level_1': False,
                 'level_2': False, 'normed': False}]
TRANS_PARAMS = [{'counts': True, 'sums': True, 'sign': 'negative',
                 'normed': True, 'convert_currency': True},
                {'counts': True, 'sums': False, 'sign': 'both',
                 'normed': True, 'convert_currency': False},
                {'counts': True, 'sums': False, 'sign': 'combined',
                 'normed': True, 'convert_currency': False},
                {'counts': True, 'sums': False, 'sign': 'combined',
                 'normed': True, 'convert_currency': False},
                {'counts': True, 'sums': True, 'sign': 'combined',
                 'normed': False, 'convert_currency': False}]
CLICK_TIME_FEATURES = [['hour'], ['45min'], ['hour'], ['90min'], ['hour']]
TRANS_TIME_FEATURES = [['hour'], ['45min'], ['hour_pos', 'hour_neg'], ['90min'], ['hour'] ]

CLICK_CATEGORIES_PATH = 'data/click_categories.csv'

 
def main():

    data_path, output_path = sys.argv[1:]

    df_click_list, df_trans_list = prepare_features(data_path)
    clf_list = load_models()
    submission = make_predictions(clf_list, df_click_list, df_trans_list)
    np.savez(output_path, submission)


def prepare_features(data_path):

    clickstream_data = aggregate_clickstream(f'{data_path}/clickstream.csv')
    transactions_data = aggregate_transactions(f'{data_path}/transactions.csv')
    
    click_categories = pd.read_csv(CLICK_CATEGORIES_PATH)
    click_categories.fillna('NaN', inplace=True)

    df_click_list = []
    df_trans_list = []
    for i in range(len(CLICK_PARAMS)):

        df_click = click_preprocess(
            clickstream_data[CLICK_DATA[i]],  click_categories, **CLICK_PARAMS[i])
        for feature_group in CLICK_TIME_FEATURES[i]:
            df_click = df_click.join(clickstream_data[feature_group])
        with open(f"data/click_features_{i+1}.pkl", "rb") as file_:
            click_features = pickle.load(file_)
        df_click = filter_features(df_click, click_features)
        df_click_list.append(df_click)

        df_trans = trans_preprocess(transactions_data[TRANS_DATA[i]], **TRANS_PARAMS[i])
        for feature_group in TRANS_TIME_FEATURES[i]:
            df_trans = df_trans.join(transactions_data[feature_group])
        with open(f"data/trans_features_{i+1}.pkl", "rb") as file_:
            trans_features = pickle.load(file_)
        df_trans = filter_features(df_trans, trans_features)
        df_trans_list.append(df_trans)

        print('df_click', df_click.shape, 'df_trans', df_trans.shape)

    return df_click_list, df_trans_list


def load_models():

    clf_list = []
    for i in range(len(RANK)):
        if RANK[i]:
            clf = CatBoostRanker()
        else:
            clf = CatBoostClassifier()
        clf.load_model(f'data/model_{i+1}.cbm')
        clf_list.append(clf)

    return clf_list


def make_predictions(clf_list, df_click_list, df_trans_list):

    list_of_rtk = list(df_click_list[0].index)
    list_of_bank= list(df_trans_list[0].index)

    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)

    batch_size = 20
    num_of_batches = int((len(list_of_bank))/batch_size)+1
    final_submission = []

    for n in range(num_of_batches):

        bank_ids = list_of_bank[(n*batch_size):((n+1)*batch_size)]

        if len(bank_ids) != 0:

            submission_part = submission[submission.bank.isin(bank_ids)].explode('rtk')

            probas = []
            for i in range(len(clf_list)):
                submission_part_i = submission_part \
                    .merge(df_trans_list[i], how='left', left_on='bank', right_index=True) \
                    .merge(df_click_list[i], how='left', left_on='rtk', right_index=True)

                Xtest = submission_part_i.drop(['bank', 'rtk'], axis=1).values
                if RANK[i]:
                    proba = clf_list[i].predict(Xtest)
                else:
                    proba = clf_list[i].predict_proba(Xtest)[:, 1]
                probas.append(proba)

            probas = np.array(probas)
            if WEIGHTS is None:
                submission_part['proba'] = np.mean(probas, axis=0)
            else:
                submission_part['proba'] = (probas.T * WEIGHTS).sum(axis=1)

            zeros_part = pd.DataFrame(bank_ids, columns=['bank'])
            zeros_part['rtk'] = 0.
            zeros_part['proba'] = 1000

            submission_part = pd.concat((submission_part, zeros_part))

            submission_part = submission_part.sort_values(
                by=['bank', 'proba'], ascending=False).reset_index(drop=True)
            submission_part = submission_part.pivot_table(index='bank', values='rtk', aggfunc=list)
            submission_part['rtk'] = submission_part['rtk'].apply(lambda x: x[:100])
            submission_part = submission_part.reset_index()
            final_submission.append(submission_part)

    final_submission = pd.concat(final_submission)
    final_submission = np.array(final_submission, dtype=object)
    print(final_submission.shape)

    return final_submission


if __name__ == "__main__":

    main()
