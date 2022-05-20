import gc

import pandas as pd


def aggregate_transactions(transactions_path):

    transactions = pd.read_csv(transactions_path,
                               dtype={'mcc_code': 'int16',
                                      'currency_rk': 'int16',
                                      'transaction_amt': 'float32'})
    
    transactions['sign'] = (transactions.transaction_amt > 0).map({
        True: 'positive', False: 'negative'})
    transactions['datetime'] = pd.to_datetime(transactions.transaction_dttm)
    transactions['week'] = transactions.datetime.dt.weekofyear
    transactions['date'] = transactions.datetime.dt.date
    transactions['hour'] = transactions.datetime.dt.hour

    transactions['minute'] = transactions.datetime.dt.minute
    transactions['minute'] = transactions['hour'] * 60 + transactions['minute']
    transactions['45min'] = transactions.minute // 45
    transactions['90min'] = transactions.minute // 90

    transactions_grouped = transactions.groupby(['user_id', 'mcc_code', 'currency_rk', 'sign']) \
        .transaction_amt.agg(['count', 'sum', 'mean'])
    transactions_grouped = transactions_grouped.reset_index()

    transactions_week = transactions.drop_duplicates(['user_id', 'mcc_code', 'sign', 'week'])
    transactions_grouped_week = transactions_week.groupby(['user_id', 'mcc_code', 'sign']).size()
    transactions_grouped_week = transactions_grouped_week.rename('count')
    transactions_grouped_week = transactions_grouped_week.reset_index()

    transactions_date = transactions.drop_duplicates(['user_id', 'mcc_code', 'sign', 'date'])
    transactions_grouped_date = transactions_date.groupby(['user_id', 'mcc_code', 'sign']).size()
    transactions_grouped_date = transactions_grouped_date.rename('count')
    transactions_grouped_date = transactions_grouped_date.reset_index()

    transactions_weekly_normed = agg_trans_weekly_normed(transactions)

    transactions_hour = time_features(transactions, column='hour',
                                      prefix='trans_hour', add_total_count=True)
    transactions_hour_neg = time_features(transactions[transactions.sign == 'negative'],
                                          column='hour', prefix='trans_neg_hour',
                                          add_total_count=True)
    transactions_hour_pos = time_features(transactions[transactions.sign == 'positive'],
                                          column='hour', prefix='trans_pos_hour',
                                          add_total_count=True)
    transactions_45min = time_features(transactions, column='45min',
                                       prefix='trans_45min', add_total_count=True) 
    transactions_90min = time_features(transactions, column='90min',
                                       prefix='trans_90min', add_total_count=True)                                   

    del transactions, transactions_week, transactions_date
    gc.collect()

    return {'grouped': transactions_grouped,
            'grouped_week': transactions_grouped_week,
            'grouped_date': transactions_grouped_date,
            'weekly_normed': transactions_weekly_normed,
            'hour': transactions_hour,
            'hour_neg': transactions_hour_neg,
            'hour_pos': transactions_hour_pos,
            '45min': transactions_45min,
            '90min': transactions_90min}


def agg_trans_weekly_normed(transactions):

    total_num_weeks = transactions.week.nunique()
    transactions_by_week = transactions.groupby(['user_id', 'mcc_code', 'sign', 'week']) \
        .transaction_amt.agg(['count', 'sum'])

    counts = transactions_by_week['count'].unstack(level='mcc_code')
    counts = counts.divide(counts.sum(axis=1), axis=0)
    counts = counts.stack()
    counts = counts.reset_index().groupby(['user_id', 'mcc_code', 'sign'])[0].sum()/total_num_weeks
    counts = counts.rename('count').to_frame()

    sums = transactions_by_week['sum'].unstack(level='mcc_code')
    sums = sums.divide(sums.sum(axis=1), axis=0)
    sums = sums.stack()
    sums = sums.reset_index().groupby(['user_id', 'mcc_code', 'sign'])[0].sum()/total_num_weeks
    sums = sums.rename('sum')

    transactions_weekly_normed = counts.join(sums).reset_index()

    return transactions_weekly_normed


def aggregate_clickstream(clickstream_path):

    clickstream = pd.read_csv(clickstream_path,
                              dtype={'cat_id': 'int16', 'new_uid': 'int32'})

    clickstream['datetime'] = pd.to_datetime(clickstream.timestamp)
    clickstream['week'] = clickstream.datetime.dt.weekofyear
    clickstream['date'] = clickstream.datetime.dt.date
    clickstream['hour'] = clickstream.datetime.dt.hour

    clickstream['minute'] = clickstream.datetime.dt.minute
    clickstream['minute'] = clickstream['hour'] * 60 + clickstream['minute']
    clickstream['45min'] = clickstream.minute // 45
    clickstream['90min'] = clickstream.minute // 90

    clickstream_week = clickstream.drop_duplicates(['user_id', 'cat_id', 'week'])
    clickstream_grouped_week = clickstream_week.groupby(['user_id', 'cat_id']).size()
    clickstream_grouped_week = clickstream_grouped_week.reset_index()

    clickstream_date = clickstream.drop_duplicates(['user_id', 'cat_id', 'date'])
    clickstream_grouped_date = clickstream_date.groupby(['user_id', 'cat_id']).size()
    clickstream_grouped_date = clickstream_grouped_date.reset_index()

    clickstream_weekly_normed = agg_click_weekly_normed(clickstream_date)

    clickstream_hour = time_features(clickstream, column='hour',
                                     prefix='click_hour', add_total_count=True)
    clickstream_45min = time_features(clickstream, column='45min',
                                      prefix='click_45min', add_total_count=True)
    clickstream_90min = time_features(clickstream, column='90min',
                                      prefix='click_90min', add_total_count=True)

    del clickstream, clickstream_week, clickstream_date
    gc.collect()

    return {'grouped_week': clickstream_grouped_week.reset_index(),
            'grouped_date': clickstream_grouped_date,
            'weekly_normed': clickstream_weekly_normed,
            'hour': clickstream_hour,
            '45min': clickstream_45min,
            '90min': clickstream_90min}


def agg_click_weekly_normed(clickstream_date):

    total_num_weeks = clickstream_date.week.nunique()
    clickstream_by_week = clickstream_date.groupby(['user_id', 'cat_id', 'week']).size()

    counts = clickstream_by_week.unstack(level='cat_id')
    counts = counts.divide(counts.sum(axis=1), axis=0)
    counts = counts.stack()
    counts = counts.reset_index().groupby(['user_id', 'cat_id'])[0].sum()/total_num_weeks

    return counts.reset_index()


def time_features(df, column='hour', prefix='trans_hour', add_total_count=True):

    df = df.drop_duplicates(['user_id', column, 'date'])
    res = df.groupby(['user_id', column]).size().unstack().fillna(0)
    res.columns = res.columns.map(lambda x: f'{prefix}_{x}')
    total_count = res.sum(axis=1)
    res = res.divide(total_count, axis=0)
    if add_total_count:
        res[f'{prefix}_total'] = total_count
    
    return res
