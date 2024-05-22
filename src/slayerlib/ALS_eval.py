#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession
import time
from ALS_conf import conf
from ALS import predict, evaluate, compute_gt


def main(spark, userID, params):
    '''READ DATA'''
    params = eval(params)
    if conf.TRAIN_KIND == 'train_small':
        pred = spark.read.parquet(
            'hdfs:/user/{}/asl-{}-{}-{}_pred_small.parquet'.format(userID, params['rank'], params['regParam'], params['alpha'])).repartition('user_id').cache()
        train_gt = spark.read.parquet('hdfs:/user/{}/asl-train_gt_small.parquet'.format(
            userID)).repartition('user_id').cache()
        val_gt = spark.read.parquet(
            'hdfs:/user/{}/asl-val_gt_small.parquet'.format(userID)).repartition('user_id').cache()
    else:
        pred = spark.read.parquet(
            'hdfs:/user/{}/asl-{}-{}-{}_pred.parquet'.format(userID, params['rank'], params['regParam'], params['alpha'])).repartition('user_id').cache()
        train_gt = spark.read.parquet(
            'hdfs:/user/{}/asl-train_gt.parquet'.format(userID)).repartition('user_id').cache()
        val_gt = spark.read.parquet(
            'hdfs:/user/{}/asl-val_gt.parquet'.format(userID)).repartition('user_id').cache()

    ''' EVALUATE'''
    print("Evaluating...")
    s = time.time()
    t_map = evaluate(pred, train_gt)
    v_map = evaluate(pred, val_gt)
    e = time.time()
    print("time: {}s".format(e - s))
    print("Train mAP: {}".format(t_map))
    print("Val mAP: {}".format(v_map))


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').config(
        "spark.sql.broadcastTimeout", 20 * 60).config("spark.sql.autoBroadcastJoinThreshold", -1).getOrCreate()
    userID = os.environ['USER']
    params = sys.argv[1]
    main(spark, userID, params)
