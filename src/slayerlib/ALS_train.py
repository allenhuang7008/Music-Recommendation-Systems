#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession
import time
from ALS_conf import conf
from ALS import predict, evaluate
from pyspark.ml.feature import StringIndexer


def main(spark, userID, params):
    '''READ DATA'''
    if conf.TRAIN_KIND == 'train_small':
        train = spark.read.parquet(
            'hdfs:/user/{}/asl-train_small.parquet'.format(userID)).cache()
        val_gt = spark.read.parquet(
            'hdfs:/user/{}/asl-val_gt_small.parquet'.format(userID)).repartition('user_id').cache()
    else:
        train = spark.read.parquet(
            'hdfs:/user/{}/asl-train.parquet'.format(userID)).cache()
        val_gt = spark.read.parquet(
            'hdfs:/user/{}/asl-val_gt.parquet'.format(userID)).repartition('user_id').cache()

    '''TRAIN'''
    print("Training...")
    s = time.time()
    pred = predict(train, params, userID)
    e = time.time()
    print("time: {}s".format(e - s))

    '''REPARTITION'''
    s = time.time()
    print("Repartition...")
    val_gt = val_gt.repartition(50, 'user_id').cache()
    pred = pred.repartition(50, 'user_id').cache()
    test_gt = spark.read.parquet(
        'hdfs:/user/{}/asl-test_gt.parquet'.format(userID)).repartition('user_id').repartition(50, 'user_id').cache()
    e = time.time()
    print("time: {}s".format(e - s))

    ''' EVALUATE'''
    print("Evaluating...")
    s = time.time()
    v_map = evaluate(pred, val_gt)
    t_map = evaluate(pred, test_gt)
    e = time.time()
    print("Val mAP: {}".format(v_map))
    print("Test mAP: {}".format(t_map))
    print("time: {}s".format(e - s))

    '''WRITE PREDICTIONS'''
    params = eval(params)
    if conf.TRAIN_KIND == 'train_small':
        pred.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-{}-{}-{}_pred_small.parquet'.format(userID, params['rank'], params['regParam'], params['alpha']))
    else:
        pred.write.mode('overwrite').parquet(
            'hdfs:/user/{}/asl-{}-{}-{}_pred.parquet'.format(userID, params['rank'], params['regParam'], params['alpha']))


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').config(
        "spark.sql.broadcastTimeout", 20 * 60).config("spark.sql.autoBroadcastJoinThreshold", -1).getOrCreate()
    userID = os.environ['USER']
    params = sys.argv[1]
    main(spark, userID, params)
