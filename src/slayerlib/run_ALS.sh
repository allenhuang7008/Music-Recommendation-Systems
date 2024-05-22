#!/bin/bash
ranks=$(python -c 'from ALS_conf import conf; print(" ".join(str(i) for i in conf.ALS_PARAM["rank"]))')
regParams=$(python -c 'from ALS_conf import conf; print(" ".join(str(i) for i in conf.ALS_PARAM["regParam"]))')
alphas=$(python -c 'from ALS_conf import conf; print(" ".join(str(i) for i in conf.ALS_PARAM["alpha"]))')

declare -a RANK_SEARCH_SPACE=($ranks)
declare -a REGPARAM_SEARCH_SPACE=($regParams)
declare -a ALPHA_SEARCH_SPACE=($alphas)
FILENAME="ALS_results_small.txt"
ATTEMP=0
touch $FILENAME
for rank in "${RANK_SEARCH_SPACE[@]}"
do
    for regParam in "${REGPARAM_SEARCH_SPACE[@]}"
    do
        for alpha in "${ALPHA_SEARCH_SPACE[@]}"
        do
            echo "Rank: $rank, regParam: $regParam, alpha: $alpha" >> $FILENAME
            while true; do
                ATTEMP=$((ATTEMP+1))
                echo "Attempt: $ATTEMP" >> $FILENAME
                spark-submit --deploy-mode client ALS_train.py "{'rank': ${rank}, 'regParam': ${regParam}, 'alpha': ${alpha}}" >> $FILENAME
                # if exit code is control c or 0 then exit
                if [ $? -eq 0 ] || [ $? -eq 130 ]; then
                    break
                fi
                sleep 5
            done
        done
    done
done