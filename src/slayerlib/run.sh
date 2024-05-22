#!/bin/bash
str=$(python -c 'from conf import conf; print(" ".join(str(i) for i in conf.BETA_SEARCH_SPACE))')
declare -a BETA_SEARCH_SPACE=($str)

spark-submit --deploy-mode client preprocess.py

for beta in "${BETA_SEARCH_SPACE[@]}"
do
    echo "beta=$beta"
    spark-submit --deploy-mode client train.py $beta
done
spark-submit --deploy-mode client eval.py