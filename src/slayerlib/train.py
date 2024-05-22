#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession

from conf import conf


def main(spark, userID, beta):
    '''READ DATA'''
    train = spark.read.parquet(
        conf.PQ_PREFIX.format(userID, conf.TRAIN_KIND)).cache()

    '''BUILD POPULARITY MODEL'''
    pred = baseline_fit(train, False, beta, conf.PERCENTILE_CUT)
    pred_vote = baseline_fit(train, True, beta, conf.PERCENTILE_CUT)

    '''SAVE PARQUET'''
    size = ''
    if conf.TRAIN_KIND == 'train_small':
        size = '-small'
    pred.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, "pred-baseline"+size))
    pred_vote.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, "pred_vote-{}".format(str(beta))+size))


def baseline_fit(df, vote=False, beta=10, perc=0.05):
    # calculate top100
    df.createOrReplaceTempView('df')
    if not vote:
        pred = spark.sql('''
        SELECT collect_list(recording_msid) AS pred
        FROM(
            SELECT df.recording_msid
            FROM df
            GROUP BY df.recording_msid
            ORDER BY count(*) DESC
            LIMIT 100
        )
        ''')
    else:
        spark.sql('''
        select recording_msid, count(*) as play_count
            from df
            group by recording_msid
        ''').createOrReplaceTempView('msid_count')
        threshold = spark.sql('''
        SELECT percentile(play_count, {})
        FROM msid_count
        '''.format(str(perc))).collect()[0][0]
        # filter out msid with play_count < threshold
        df = spark.sql('''
        select *
        from df
        where recording_msid in (
            select recording_msid
            from msid_count
            where play_count >= {}
        )
        '''.format(str(threshold)))
        df.createOrReplaceTempView('df')
        pred = spark.sql('''
        with majority as (
        select user_id, recording_msid, row_number() over(partition by user_id order by interaction_count desc) as rank
        from df
        ),
        temp as(
        select recording_msid, (101-rank) as rating
        from majority
        where rank <= 100
        ),
        temp2 as(
        select recording_msid, sum(rating)/(count(*)+{}) as score
        from temp
        group by recording_msid
        order by score desc
        limit 100)
        select collect_list(recording_msid) AS pred, collect_list(score) AS slice_rank
        from temp2
        '''.format(str(beta)))
    return pred


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').getOrCreate()
    userID = os.environ['USER']
    beta = int(sys.argv[-1])
    main(spark, userID, beta)
