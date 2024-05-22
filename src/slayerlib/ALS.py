from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import transform, col, collect_list, udf
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, lit, explode, udf
from pyspark.ml.evaluation import RankingEvaluator

def predict(train_data, params, userID):
    '''
      params: a system input dictionary with keys [rank, maxIter, regParam] in a string form
    '''
    als = ALS(
        maxIter=1,
        userCol="user_index",
        itemCol="track_index",
        ratingCol="score",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=True
    )
    als.setParams(**eval(params))

    # reindex user_id and track_id
    print("reindexing...")
    w = Window().partitionBy("user_id").orderBy("user_id")
    train_data = train_data.withColumn("user_index", row_number().over(w))
    user_map = train_data.select(['user_index', 'user_id']).distinct()
    train_data = train_data.drop('user_id')
    w = Window().partitionBy("track_id").orderBy("track_id")
    train_data = train_data.withColumn("track_index", row_number().over(w))
    track_map = train_data.select(['track_index', 'track_id']).distinct()
    train_data = train_data.drop('track_id')

    print("fitting model...")
    model = als.fit(train_data)
    # col[user_id, [(track_id, score), (track_id, score), ...]] orderd by score asc.
    pred = model.recommendForAllUsers(numItems=100)
    # Flatten the pred to col[user_id, [track_id, track_id, ...]]

    print("transforming...")
    pred = pred.select("user_index", transform(
        col("recommendations"), lambda rec: rec.track_index).alias("track_index"))
    exploded_df = pred.select('user_index', explode(
        'track_index').alias('track_index'))
    pred = exploded_df.join(user_map, on='user_index',
                            how='inner').drop("user_index")
    pred = pred.join(track_map, on='track_index', how='inner')
    pred = pred.withColumnRenamed(
        'track_id', 'pred').select(["user_id", "pred"])
    pred = pred.groupBy('user_id').agg(collect_list('pred').alias('pred'))
    user_map = user_map.withColumnRenamed('user_index', 'id')
    track_map = track_map.withColumnRenamed('track_index', 'id')
    matrix_u = model.userFactors.join(user_map, on='id', how='inner')
    matrix_i = model.itemFactors.join(track_map, on='id', how='inner')

    print("write factors...")
    # user_map.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-user_map.parquet'.format(userID))
    # track_map.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-track_map.parquet'.format(userID))
    # pred.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-pred.parquet'.format(userID))
    # matrix_i.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-matrix_i.parquet'.format(userID))
    # matrix_u.write.mode('overwrite').parquet(
    #     'hdfs:/user/{}/asl-matrix_u.parquet'.format(userID))
    return pred


def evaluate(pred, gt):
    merge = pred.join(gt, on='user_id', how='inner')
    rdd = merge.select('pred', 'gt').rdd.cache()
    metrics = RankingMetrics(rdd)
    mean_ap = metrics.meanAveragePrecision
    return mean_ap

def evaluate_2(pred, gt):
    udf_func = udf(lambda x: [float(i) for i in x], ArrayType(FloatType()))
    pred = pred.withColumn('track_id', udf_func('track_id')).withColumnRenamed('track_id', 'rec')
    gt = gt.withColumn('track_id', udf_func('track_id')).withColumnRenamed('track_id', 'label')
    df = pred.join(gt, pred.user_id==gt.user_id)
    evaluator = RankingEvaluator(predictionCol='rec', labelCol='label', metricName='meanAveragePrecision')
    mean_AP = evaluator.evaluate(df)
    return mean_AP


def compute_gt(df):
    w = Window().partitionBy("user_id").orderBy(col("score").desc())
    df = df.withColumn("rank", row_number().over(w)).filter(
        col("rank") <= 100).drop("rank")
    gt = df.groupBy('user_id').agg(collect_list('track_id').alias('gt'))
    return gt


def encode_id(df, id_col):
    return df
