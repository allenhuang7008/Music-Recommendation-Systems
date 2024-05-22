import os
import sys
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics

from conf import conf


def main(spark, userID):
    '''READ DATA'''
    val_name = 'val'
    if conf.TRAIN_KIND == 'train_small':
        val_name = 'val_small'

    train_gt = spark.read.parquet(conf.PQ_PREFIX.format(
        userID, conf.TRAIN_KIND+"_gt")).cache()
    val_gt = spark.read.parquet(conf.PQ_PREFIX.format(
        userID, val_name+"_gt")).cache()
    test_gt = spark.read.parquet(conf.PQ_PREFIX.format(
        userID, "test_gt")).cache()

    '''EVALAUTE ON VAL'''
    size = ''
    if conf.TRAIN_KIND == 'train_small':
        size = '-small'
    pred = spark.read.parquet(conf.PQ_PREFIX.format(
        userID, "pred-baseline"+size)).cache()
    print("Baseline")
    print("Train Mean Average Precision = ", evaluate(pred, train_gt))
    print("Val Mean Average Precision = ", evaluate(pred, val_gt))
    print("Test Mean Average Precision = ", evaluate(pred, test_gt))

    best_beta = 0
    metrics = dict(train=0, val=0, test=0)
    for beta in conf.BETA_SEARCH_SPACE:
        pred_vote = spark.read.parquet(conf.PQ_PREFIX.format(
            userID, "pred_vote-{}".format(str(beta))+size)).cache()
        vote_eval_train, vote_eval_val = evaluate(
            pred_vote, train_gt), evaluate(pred_vote, val_gt)
        if vote_eval_val > metrics['val']:
            best_beta, opt_pred, max_prec = beta, pred_vote, vote_eval_val
        print("Beta = {}".format(str(beta)))
        print("Train Mean Average Precision = ", vote_eval_train)
        print("Val Mean Average Precision = ", vote_eval_val)

    '''EVALUATE OPT PRED ON TEST'''
    print("="*10, "final output", "="*10)
    print("optimal beta = {}".format(str(best_beta)))
    print("Test Mean Average Precision = ", evaluate(opt_pred, test_gt))


def evaluate(pred, gt):
    merge = gt.crossJoin(pred)
    rdd = merge.select('pred', 'gt').rdd.map(lambda row: (row.pred, row.gt))
    metrics = RankingMetrics(rdd)
    mean_ap = metrics.meanAveragePrecision
    return mean_ap


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').getOrCreate()
    userID = os.environ['USER']
    main(spark, userID)
