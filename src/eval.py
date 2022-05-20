"""
Make predictions and calculate metrics.
"""

import pandas as pd
from tqdm import tqdm


def make_predictions(clf, test, df_trans, df_click, batch_size=20):
    
    preds = test.bank.to_frame().reset_index(drop=True)
    preds['rtk'] = preds.bank.apply(lambda x: test.rtk.tolist())
    
    num_of_batches = int((len(preds))/batch_size)+1
    final_preds = []

    with tqdm(range(num_of_batches)) as pbar:
        for i in pbar:

            bank_ids = preds.bank[(i*batch_size):((i+1)*batch_size)]

            preds_part = preds[preds.bank.isin(bank_ids)].explode('rtk')
            preds_part = preds_part.merge(df_trans, how='left', left_on='bank', right_index=True) \
                .merge(df_click, how='left', left_on='rtk', right_index=True)

            preds_part = preds_part.sort_values('bank')
            X = preds_part.drop(['bank', 'rtk'], axis=1).values

            preds_part['proba'] = clf.predict(X)
            preds_part = preds_part[['bank', 'rtk', 'proba']]

            preds_part = preds_part.sort_values(
                by=['bank', 'proba'],ascending=False).reset_index(drop=True)
            preds_part = preds_part.pivot_table(index='bank', values='rtk', aggfunc=list)
            preds_part['rtk'] = preds_part['rtk'].apply(lambda x: x[:100])
            preds_part = preds_part.reset_index()
            final_preds.append(preds_part)

            current_preds = pd.concat(final_preds)
            r1, mrr, precision = calc_metrics(current_preds, test)
            pbar.set_postfix(r1=r1)

    return pd.concat(final_preds)


def calc_metrics(preds, test):
    
    final = pd.merge(preds, test.rename(columns={'rtk': 'ytrue'}))
    final['precision'] = final.apply(apply_precision, axis=1)
    final['mrr'] = final.apply(apply_mrr, axis=1)
    
    precision = final.precision.mean()
    mrr = final.mrr.mean()
    r1 = calc_r1(precision, mrr)

    return r1, mrr, precision


def apply_precision(row):
    
    if row.ytrue in row.rtk[:100]:
        return 1
    else:
        return 0


def apply_mrr(row):
    
    try:
        mrr = 1/(row.rtk[:100].index(row.ytrue) + 1)
    except ValueError:
        mrr = 0
    
    return mrr


def calc_r1(precision, mrr):
    
    try:
        return 2 * precision * mrr / (precision + mrr)
    except ZeroDivisionError:
        return 0
