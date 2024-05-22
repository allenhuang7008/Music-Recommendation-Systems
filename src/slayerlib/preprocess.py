#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession

from helper import read, get_thres, show_stats
from conf import conf

path = 'hdfs:/user/bm106_nyu_edu/1004-project-2023'


def main(spark, userID):
    '''READ DATA'''
    inter, _, _ = read(spark, path, conf.TRAIN_KIND)
    test_inter, _, _ = read(spark, path, "test")

    '''STATS'''
    show_stats(spark, inter)

    '''TRAIN/VAL SPLIT'''
    train, val = preprocess(inter, conf.PERCENTILE_CUT, False)
    test = preprocess(test_inter, conf.PERCENTILE_CUT, True)
    print("Train", (train.count(), len(train.columns)))
    print("Val", (val.count(), len(val.columns)))
    print("Test", (test.count(), len(test.columns)))
    '''COMPUTE GROUND TRUTH'''
    print("Computing ground truth...")
    train_gt = compute_gt(train, True)
    val_gt = compute_gt(val, True)
    test_gt = compute_gt(test, True)

    '''SAVE PARQUET'''
    print("Saving parquet...")
    train.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, conf.TRAIN_KIND))

    val_name = 'val'
    if conf.TRAIN_KIND == 'train_small':
        val_name = 'val_small'

    train_gt.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, conf.TRAIN_KIND+"_gt"))
    val_gt.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, val_name+"_gt"))
    test_gt.write.mode('overwrite').parquet(
        conf.PQ_PREFIX.format(userID, "test_gt"))


def preprocess(inter, perc_cut, test=False):
    thres = get_thres(spark, inter, perc_cut)
    inter.createOrReplaceTempView('inter')
    grouped = spark.sql('''
    with p1 as (
    select user_id, recording_msid, count(*) as interaction_count, count(*) over(partition by user_id) as track_count, ntile(5) over(partition by user_id order by rand(10)) as group
    from inter
    group by user_id, recording_msid
    order by user_id
    )
    select *
    from p1
    where track_count >= {}
    '''.format(str(thres)))
    grouped.createOrReplaceTempView('grouped')
    if not test:
        # train interaction table
        train = spark.sql('''
        select user_id, recording_msid, interaction_count
        from grouped
        where group < 5
        ''')

        # val interaction table
        val = spark.sql('''
        select user_id, recording_msid, interaction_count
        from grouped
        where group = 5
        ''')
        return train, val
    else:
        # test interaction table
        test = spark.sql('''
        select user_id, recording_msid, interaction_count
        from grouped
        ''')
        return test


def compute_gt(df, vote=False):
    df.createOrReplaceTempView('df')
    # calculate val100
    if not vote:
        gt = spark.sql('''
        SELECT user_id, SLICE(collect_list(recording_msid), 1, 100) AS gt
        FROM (
        SELECT user_id, recording_msid, COUNT(*) as count
        FROM df
        GROUP BY user_id, recording_msid
        ORDER BY user_id, count DESC
        )
        GROUP BY user_id
        ORDER BY user_id
        ''')
    else:
        gt = spark.sql('''
        with majority as (
        select user_id, recording_msid, row_number() over(partition by user_id order by interaction_count desc) as rank
        from df
        order by user_id, rank asc
        )
        select user_id, collect_list(recording_msid) as gt, collect_list(rank) AS slice_rank
        from majority
        where rank <= 100
        group by user_id
        order by user_id
        ''')
    return gt


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').getOrCreate()
    userID = os.environ['USER']
    main(spark, userID)
