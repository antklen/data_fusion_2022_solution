# Data Fusion 2022 Contest

2nd place solution for [Data Fusion 2022 Contest](https://ods.ai/competitions/data-fusion2022-main-challenge).


## Approach

Main part of solution is different feature engineering:
- Use time features. For each user and hour count number of distinct days when user had clicks/transactions. This works better than naive approach with counting number of clicks/transactions.
- Calculate counts for each user and click category, counts and sums for each user and mcc code of transaction. Again helps to count number of distinct days/weeks for each user-category pair instead of simply counting number of clicks/transactions.
- Normalize features by user, e.g. divide counts (or sums) by total count (total sum) for given user.
- Filtering out rare categories and mcc codes.

Training:
- CatBoostRanker with YetiRank loss.
- Training for long time - 15000 iterations.
- To train we need to sample negative examples. We sample only small fraction of all possible negative pairs. So we can add resampling after each 1000 iterations so model will see more diverse data. This helps a little.

Ensemble:
- Simple average of 5 models with different feature engineering and the same catboost parameters.

## Details

To reproduce solution competition data should be added to `data` folder:
- `clickstream.csv`
- `transactions.csv`
- `click_categories.csv`
- `train_matching.csv`

`run_all.sh` contains all steps to fully reproduce solution:
- `python src/aggregate.py` - aggregate raw data for further feature engineering.
- `python src/run_training.py --config-name=run{1,2,3,4,5}` - training 5 different models.

`submit` folder contains final submission. Trained models will be automatically added to it.