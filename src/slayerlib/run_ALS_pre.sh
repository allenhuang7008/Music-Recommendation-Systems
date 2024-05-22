#!/bin/bash
# spark-submit --deploy-mode client gen_id.py
FILENAME="ALS_results_small.txt"
touch $FILENAME
echo "Preprocessing..." >> $FILENAME
spark-submit --deploy-mode client ALS_preprocess.py >> $FILENAME