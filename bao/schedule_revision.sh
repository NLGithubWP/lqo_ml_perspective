#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# 5 arm,
cd /app/bao/bao_server
nohup python3 -u main.py >./server_train_bao_imdb_ori_query_full_test.log 2>&1 &

# (imdb_13, JOB_full train) + (imdb_13, JOB_full test)
cd /app/bao
python3 run_queries.py --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/train_bao_imdb_ori_query_full.log --database_name imdb_ori
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_bao_imdb_ori_query_full.log --database_name imdb_ori

# (imdb_13, JOB_full train) + (imdb_13_d_0.1, JOB_full test)
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_bao_imdb_01v2_query_full.log --database_name imdb_01v2

# (imdb_13, JOB_full train) + (imdb_13_d_0.5, JOB_full test)
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_bao_imdb_05v2_query_full.log --database_name imdb_05v2

# (imdb_13, JOB_full train) + (imdb_17, JOB_full test)
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_bao_imdb_17_query_full.log --database_name imdb_17v2

# clean and save the model
cd /app/bao/bao_server
mv bao_default_model bao_default_model_imdb_ori_query_full
mv bao_previous_model bao_previous_model_imdb_ori_query_full
rm bao.db
# pkill -f "python3 -u main.py"
