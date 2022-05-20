"""
Sample negative examples for model training.
"""

import numpy as np
import pandas as pd


def sample_negative_examples(data, sample_size=10, random_state=42):

    bank_sample = pd.concat([data.bank] * sample_size).sample(
            frac=1, replace=False, random_state=random_state).reset_index(drop=True)
    rtk = data.rtk[data.rtk != '0']
    size_ratio = len(data) / len(rtk)
    rtk_sample = pd.concat([rtk] * int(np.ceil(sample_size * size_ratio))).sample(
            frac=1, replace=False, random_state=random_state + 1).reset_index(drop=True)
    sample = pd.DataFrame({'bank': bank_sample,
                            'rtk': rtk_sample.iloc[:len(bank_sample)],
                            'target': 0})
            
    final_df = pd.concat([data, sample])
    final_df = final_df[final_df.rtk != '0']
    final_df = final_df.groupby(['bank', 'rtk']).target.max().reset_index()
    final_df = final_df.sample(frac=1, replace=False, random_state=random_state)
    
    return final_df
