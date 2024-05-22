def read(spark, path, postfix):
    inter = spark.read.parquet(
        '{}/interactions_{}.parquet'.format(path, postfix)).cache()
    track = spark.read.parquet(
        '{}/tracks_{}.parquet'.format(path, postfix)).cache()
    user = spark.read.parquet(
        '{}/users_{}.parquet'.format(path, postfix)).cache()
    return inter, track, user


def show_stats(spark, inter):
    inter.createOrReplaceTempView('inter')
    stats = spark.sql('''
    SELECT MIN(track_count) AS MinTrackCount
    ,percentile(track_count, 0.25) AS Q1
    ,percentile(track_count, 0.50) AS Q2
    ,percentile(track_count, 0.75) AS Q3
    ,MAX(track_count) AS MaxTrackCount
    ,AVG(track_count) AS AvgRate
    FROM (
        SELECT user_id, COUNT(DISTINCT(recording_msid)) AS track_count
        FROM inter
        GROUP BY user_id
    ) 
    ''')
    stats.createOrReplaceTempView('stats')
    stats.show()


def get_thres(spark, inter, perc):
    inter.createOrReplaceTempView('inter')
    stats = spark.sql('''
    SELECT percentile(track_count, {})
    FROM (
        SELECT user_id, COUNT(DISTINCT(track_id)) AS track_count
        FROM inter
        GROUP BY user_id
    ) 
    '''.format(str(perc)))
    return stats.collect()[0][0]
