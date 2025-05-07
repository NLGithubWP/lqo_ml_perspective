#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Default PG on (imdb_13_d_0.1, JOB_full test)
cd /app/bao
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_pg_imdb_01v2_query_full.log --database_name imdb_01v2



# Default PG on (imdb_13_d_0.5, JOB_full test)
cd /app/bao
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_pg_imdb_05v2_query_full.log --database_name imdb_05v2


# Default PG on (imdb_17, JOB_full test)
cd /app/bao
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb/test_pg_imdb_17_query_full.log --database_name imdb_17v2
