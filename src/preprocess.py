"""
Prepare final features from transactions and clickstream aggregates.
"""

import pandas as pd


def click_preprocess(clickstream, click_categories,
                     cat_id=True, level_0=False,
                     level_1=False, level_2=False,
                     filter_count=0, normed=True):

    clicks = pd.merge(clickstream, click_categories)
    df_click = []

    if cat_id:
        df = group_clicks(clickstream, 'cat_id', filter_count, normed)
        df_click.append(df)

    if level_0:
        df = group_clicks(clicks, 'level_0', filter_count, normed)
        df_click.append(df)

    if level_1:
        df = group_clicks(clicks, 'level_1', filter_count, normed)
        df_click.append(df)

    if level_2:
        df = group_clicks(clicks, 'level_2', filter_count, normed)
        df_click.append(df)

    return pd.concat(df_click, axis=1)


def group_clicks(clicks, column='cat_id', filter_count=10, normed=False):

    df = clicks.groupby(['user_id', column])[0].sum()
    df = df.unstack().fillna(0)
    df.columns = df.columns.map(lambda x: f'{column}-{x}')
    if filter_count > 0:
        df = column_filter(df, filter_count)
    if normed:
        df = df.divide(df.sum(axis=1), axis=0)
        
    return df


def column_filter(df, filter_count):
    
    column_count = (df != 0).sum(axis=0)
    use_columns = column_count[column_count >= filter_count].index
    df = df[use_columns]
    
    return df


def trans_preprocess(transactions, counts=True, sums=False, sign='combined',
                     filter_count=10, normed=False, convert_currency=True):

    if convert_currency:
        transactions = transactions.copy()
        idx = transactions.currency_rk.isin([50, 60])
        transactions['sum'].loc[idx] = transactions['sum'].loc[idx] * 100
        transactions['mean'].loc[idx] = transactions['mean'].loc[idx] * 100

    if sign == 'combined':
        df_trans = group_counts_and_sums(transactions, counts, sums,
                                         filter_count, normed)
    elif sign == 'negative':
        transactions2 = transactions[transactions.sign == 'negative']
        df_trans = group_counts_and_sums(transactions2, counts, sums,
                                         filter_count, normed, postfix='negative')

    elif sign == 'both':
        transactions2 = transactions[transactions.sign == 'negative']
        df_trans = group_counts_and_sums(transactions2, counts, sums,
                                         filter_count, normed, postfix='negative')

        transactions2 = transactions[transactions.sign == 'positive']
        df_trans2 = group_counts_and_sums(transactions2, counts, sums,
                                          filter_count, normed, postfix='positive')
        df_trans = df_trans + df_trans2

    else:
        raise ValueError

    return pd.concat(df_trans, axis=1)


def group_counts_and_sums(transactions, counts=True, sums=False,
                          filter_count=10, normed=False, postfix=None):

    df_trans = []
    
    if counts:
        df = group_transactions(transactions, 'count', filter_count, normed)
        if postfix is not None:
            df.columns = df.columns.map(lambda x: f'{x}-{postfix}')
        df_trans.append(df)
    if sums:
        df = group_transactions(transactions, 'sum', filter_count, normed)
        if postfix is not None:
            df.columns = df.columns.map(lambda x: f'{x}-{postfix}')
        df_trans.append(df)
        
    return df_trans


def group_transactions(transactions, column='count', filter_count=10, normed=False):

    df = transactions.groupby(['user_id', 'mcc_code'])[column].sum()
    df = df.unstack().fillna(0)
    df.columns = df.columns.map(lambda x: f'{column}-mcc{x}')
    if filter_count > 0:
        df = column_filter(df, filter_count)
    if normed:
        df = df.divide(df.sum(axis=1), axis=0)
        
    return df
