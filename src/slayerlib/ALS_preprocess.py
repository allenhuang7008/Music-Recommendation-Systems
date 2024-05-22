#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession

from ALS import compute_gt
from helper import read, get_thres, show_stats
from ALS_conf import conf

path = 'hdfs:/user/bm106_nyu_edu/1004-project-2023'
NAME = 'hdfs:/user/{}/ASL-{}.parquet'


def main(spark, userID):
    '''READ DATA'''
    if conf.TRAIN_KIND == 'train_small':
        inter = spark.read.parquet(
            'hdfs:/user/{}/interactions_train_small_with_ids.parquet'.format(userID)).cache()
    else:
        inter = spark.read.parquet(
            'hdfs:/user/{}/interactions_train_with_ids.parquet'.format(userID)).cache()

    inter_test = spark.read.parquet(
        'hdfs:/user/{}/interactions_test_with_ids.parquet'.format(userID)).cache()

    '''TRAIN/VAL SPLIT'''
    train, val = preprocess(inter, conf.PERCENTILE_CUT, False)
    test = preprocess(inter_test, conf.PERCENTILE_CUT, True)
    print("Train", (train.count(), len(train.columns)))
    print("Val", (val.count(), len(val.columns)))
    # print("Test", (test.count(), len(test.columns)))

    '''COMPUTE GROUND TRUTH'''
    # repartition by user_id
    train_gt = compute_gt(train)
    val_gt = compute_gt(val)
    test_gt = compute_gt(test)

    '''SAVE PARQUET'''
    print("Saving parquet...")
    if conf.TRAIN_KIND == 'train_small':
        train.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-train_small.parquet'.format(userID))
        val.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-val_small.parquet'.format(userID))
        train_gt.write.partitionBy('user_id').mode('overwrite').parquet(
            'hdfs:/user/{}/asl-train_gt_small.parquet'.format(userID))
        val_gt.write.partitionBy('user_id').mode('overwrite').parquet(
            'hdfs:/user/{}/asl-val_gt_small.parquet'.format(userID))

    else:
        train.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-train.parquet'.format(userID))
        val.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-val.parquet'.format(userID))
        train_gt.write.partitionBy('user_id').mode('overwrite').parquet(
            'hdfs:/user/{}/asl-train_gt.parquet'.format(userID))
        val_gt.write.partitionBy('user_id').mode('overwrite').parquet(
            'hdfs:/user/{}/asl-val_gt.parquet'.format(userID))

    test_gt.write.partitionBy('user_id').mode('overwrite').parquet(
        'hdfs:/user/{}/asl-test_gt.parquet'.format(userID))
    # test.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-test.parquet'.format(userID))
    print("Done!")


def preprocess(inter, perc_cut, test=False):
    thres = get_thres(spark, inter, perc_cut)
    inter.createOrReplaceTempView('inter')
    grouped = spark.sql('''
    with p1 as (
    select user_id, track_id, count(*) as score, count(*) over(partition by user_id) as track_count, ntile(5) over(partition by user_id order by rand(10)) as group
    from inter
    group by user_id, track_id
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
        select user_id, track_id, score
        from grouped
        where group < 5
        ''')

        # val interaction table
        val = spark.sql('''
        select user_id, track_id, score
        from grouped
        where group = 5
        ''')
        return train, val
    else:
        # test interaction table
        test = spark.sql('''
        select user_id, track_id, score
        from grouped
        ''')
        return test


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').getOrCreate()
    userID = os.environ['USER']
    main(spark, userID)
