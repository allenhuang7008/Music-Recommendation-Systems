# Spark Codes

## Baseline model

0. `conf.py` specifies the parameters for the baseline model.
1. `preprocess.py` preprocesses the data for baseline model.
2. `train.py` trains the baseline model.
3. `eval.py` evaluates the baseline model.

## LFM

0. `ALS_conf.py` specifies the parameters for the ALS model.
1. The script `gen_id.py` creates the ids for user_id and track_id in both the training data and the test data. For the track, we first coalesce `recording_mbid` with `reconrding_msid` to get a unique identifier for each track. then we use row_number as the new track_id. Due to the LFM's dense index requirement, we only create ids for users and tracks that have appeared in  the train_small interaction and test interaction tables.
2. The script `ALS_pre_process.py` takes as input either the train set or the test set. If the input is the train set we do the train/validation split. The output at this step is a parquet file with 3 columns: user_id (INT), track_id (INT), score (INT). Note that the score is the interaction count, i.e. the number of times a user listened to a song.
3. The script `ALS_train.py` generates a recommendation of 100 songs for each user which includes the user_id and the prediction for each user which is a list of 100 songs.
4. The script `ALS_eval.py` takes in the recommendation list from ALS_train and compares it to the ground truth either from the validation set or test set and gives us a MAP score.

### How to Run

baseline model

```bash
chmod +x run.sh
./run.sh
```

LFM

```bash
chmod +x ./run_ALS_pre.sh run_ALS.sh
./run_ALS_pre.sh
./run_ALS.sh
```
