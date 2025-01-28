

python run_mcts.py --query_file workload/query_join__train.json --database imdb_01v2
mv model model_imdb_01v2
mv logs logs_imdb_01v2


mkdir logs
mkdir model
python run_mcts.py --query_file workload/query_join__train.json --database imdb_05v2
mv model model_imdb_05v2
mv logs logs_imdb_05v2


mkdir logs
mkdir model
python run_mcts.py --query_file workload/query_join__train.json  --database imdb_07v2
mv model model_imdb_07v2
mv logs logs_imdb_07v2

