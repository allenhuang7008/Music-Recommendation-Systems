#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, regexp_replace, lit, arrays_zip, explode, col, coalesce, row_number
from pyspark.sql.window import Window

from helper import read, get_thres, show_stats

path = 'hdfs:/user/bm106_nyu_edu/1004-project-2023'


def main(spark, userID):
    '''READ DATA'''
    inter_small, track_small, users_small = read(spark, path, 'train_small')
    inter, track, users_train = read(spark, path, 'train')
    inter_test, track_test, users_test = read(spark, path, 'test')

    '''CLEAN DATA'''
    users_small = clean_user(users_small, 'train')
    users_train = clean_user(users_train, 'train')
    users_test = clean_user(users_test, 'test')
    track_small = clean_track(track_small, 'train')
    track = clean_track(track, 'train')
    track_test = clean_track(track_test, 'test')

    '''JOIN DATA'''
    all_user = users_small.union(users_train).union(users_test)
    print("# of users: ", all_user.count())
    all_track = track.union(track_small).union(track_test)
    print("# of tracks: ", all_track.count())

    '''CREATE ID'''
    all_user_w_id = create_user_id(all_user)
    track_w_id = create_track_id(all_track)

    '''WRITE DATA'''
    all_user_w_id.write.mode('overwrite').parquet(
        'hdfs:/user/{}/{}.parquet'.format(userID, "users_with_ids"))
    track_w_id.write.mode('overwrite').parquet(
        'hdfs:/user/{}/{}.parquet'.format(userID, "tracks_with_ids"))

    # '''JOIN DATA'''
    track_id_mapping = track_w_id.select(['track_id', 'recording_msid'])
    user_id_mapping = all_user_w_id.select(['user_id', 'u_id'])

    inter_small = join_inter(inter_small, track_id_mapping, user_id_mapping)
    inter = join_inter(inter, track_id_mapping, user_id_mapping)
    inter_test = join_inter(inter_test, track_id_mapping, user_id_mapping)

    # '''WRITE DATA'''
    inter_small.write.mode('overwrite').parquet(
        'hdfs:/user/{}/{}.parquet'.format(userID, "interactions_train_small_with_ids"))
    inter.write.mode('overwrite').parquet(
        'hdfs:/user/{}/{}.parquet'.format(userID, "interactions_train_with_ids"))
    inter_test.write.mode('overwrite').parquet(
        'hdfs:/user/{}/{}.parquet'.format(userID, "interactions_test_with_ids"))


def clean_user(df, kind):
    df = df.select(
        ['user_id'])
    df = df.withColumn('type', lit(kind))
    return df


def clean_track(df, kind):
    df = df.select(
        ['recording_msid', 'recording_mbid'])
    df = df.withColumn('type', lit(kind))
    return df


def create_user_id(df):
    df = df.withColumn("dummy", lit("dum"))
    w = Window().partitionBy("dummy").orderBy(lit('A'))
    df = df.withColumn('u_id', row_number().over(w)).drop("dummy")
    return df


def create_track_id(df):
    df = df.withColumn("recording_mbid", coalesce(
        "recording_mbid", "recording_msid"))
    # rename column recording_mdbid to long_id
    df = df.withColumnRenamed("recording_mbid", "long_id")
    # remove rows with long_id euqal to empty string
    df = df.filter(df.long_id != '')
    df = df.createOrReplaceTempView('df')
    df = spark.sql('''
    SELECT long_id, collect_list(recording_msid) as msids, collect_list(type) as types
    FROM df
    GROUP BY long_id
    ''')
    df = df.withColumn("dummy", lit("dum"))
    w = Window().partitionBy("dummy").orderBy(lit('A'))
    df = df.withColumn('track_id', row_number().over(w)).drop("dummy")
    df = df.withColumn("new", arrays_zip("msids", "types"))\
        .withColumn("new", explode("new"))\
        .select(col("track_id"), col("new.msids").alias("recording_msid"), col("new.types").alias("type"))
    df = df.select(['track_id', 'recording_msid', 'type'])
    return df


def join_inter(inter, track_id_mapping, user_id_mapping):
    inter = inter.join(
        track_id_mapping, on='recording_msid', how='left').join(
        user_id_mapping, on='user_id', how='left').drop('recording_msid').drop('user_id').withColumnRenamed('u_id', 'user_id')
    return inter

# def clean_df(df):
#     clean_df = df.withColumn("col1", regexp_replace(
#         lower("track_name"), "[^a-zA-Z0-9]+", ""))
#     clean_df = clean_df.withColumn("col2", regexp_replace(
#         lower("artist_name"), "[^a-zA-Z0-9]+", ""))
#     clean_df = clean_df.withColumn(
#         "concat_col", concat_ws("", "col1", "col2"))
#     clean_df = clean_df.withColumn(
#         "concat_id", regexp_replace("concat_col", "\s+", ""))
#     clean_df.createOrReplaceTempView('clean_df')
#     clean_df = spark.sql('''
#     SELECT concat_id, collect_list(recording_msid) as msids, collect_list(artist_name) as artist_names, collect_list(track_name) as track_names, collect_list(type) as types
#     FROM clean_df
#     GROUP BY concat_id
#     ORDER BY concat_id
#     ''')
#     clean_df.show(20, False)

#     # add a monotonic increasing track_id column to the track_clean table
#     clean_df = clean_df.withColumn("track_id", monotonically_increasing_id())
#     # explode the table to get a new table with each row containing one artist_name and one track_name
#     clean_df = clean_df.withColumn("new", arrays_zip("msids", "artist_names", "track_names", "types"))\
#         .withColumn("new", explode("new"))\
#         .select(col("new.msids").alias("recording_msid"), col("new.artist_names").alias("artist_name"), col("new.track_names").alias("track_name"), col("new.types").alias("type"))
#     clean_df = clean_df.select(['track_id', 'concat_id', 'recording_msid', 'artist_name',
#                                 'track_name', 'type'])
#     return clean_df


if __name__ == "__main__":
    spark = SparkSession.builder.appName('part1').getOrCreate()
    userID = os.environ['USER']
    main(spark, userID)
