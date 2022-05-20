""""
Run model training.
"""

import pickle

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from preprocess import click_preprocess, trans_preprocess
from train import training_with_resampling


@hydra.main(config_path='config', config_name='config')
def main(config):

    print(OmegaConf.to_yaml(config))

    with open(to_absolute_path("data/transactions.pkl"), "rb") as file_:
        transactions_data = pickle.load(file_)
    with open(to_absolute_path("data/clickstream.pkl"), "rb") as file_:
        clickstream_data = pickle.load(file_)

    click_categories = pd.read_csv(to_absolute_path('data/click_categories.csv'))
    click_categories.fillna('NaN', inplace=True)

    matching = pd.read_csv(to_absolute_path('data/train_matching.csv'))
    matching['target'] = 1


    df_click = click_preprocess(clickstream_data[config.click_data], click_categories,
                                **config.click_params)
    print(df_click.shape)
    df_trans = trans_preprocess(transactions_data[config.trans_data], **config.trans_params)
    print(df_trans.shape)

    for feature_group in config.click_time_features:
        df_click = df_click.join(clickstream_data[feature_group])
    for feature_group in config.trans_time_features:
        df_trans = df_trans.join(transactions_data[feature_group])
    print('df_click', df_click.shape, 'df_trans', df_trans.shape)


    clf = training_with_resampling(
        matching, test=None, df_trans=df_trans, df_click=df_click,
        catboost_params=config.catboost_params, **config.train_params)

    clf.save_model(to_absolute_path(f'submit/data/model_{config.run_number}.cbm'))
    click_filename = f'submit/data/click_features_{config.run_number}.pkl'
    trans_filename = f'submit/data/trans_features_{config.run_number}.pkl'
    with open(to_absolute_path(click_filename), 'wb') as file_:
        pickle.dump(df_click.columns.tolist(), file_)
    with open(to_absolute_path(trans_filename), 'wb') as file_:
        pickle.dump(df_trans.columns.tolist(), file_)


if __name__ == '__main__':

    main()
