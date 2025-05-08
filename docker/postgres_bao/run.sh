#!/bin/bash

cd ~/AI4QueryOptimizer/ || { echo "Failed to cd into target directory"; exit 1; }
cd ~/datasets || { echo "Failed to cd into target directory"; exit 1; }
cd ~/pgdata || { echo "Failed to cd into target directory"; exit 1; }

# build the network
docker network create \
  --driver bridge \
  --subnet 10.5.0.0/16 \
  network


# run the postgresql docker
docker run -d \
  --name pg_bao \
  --network network \
  --ip 10.5.0.5 \
  -e PGDATA=/pgdata \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v ~/AI4QueryOptimizer/baseline/lqo_ml_perspective/bao:/app/bao \
  -v ~/datasets:/data/datasets \
  -v ~/AI4QueryOptimizer/baseline/lqo_ml_perspective/conf/bao-postgresql.conf:/app/postgresql.conf \
  -v ~/pgdata:/pgdata \
  -p 5432:5432 \
  --shm-size=32g \
  pg_bao_img


# install the PG_BAO extension + update config + restart PostgreSQL inside the container
docker exec -u postgres pg_bao bash -c "
  cd /app/bao/pg_extension &&
  make USE_PGXS=1 install &&
  cp /app/postgresql.conf /pgdata/postgresql.conf &&
  pg_ctl restart
"

docker start pg_bao

# run the server docker
docker run --gpus all -d \
  --name bao_server \
  --network network \
  --ip 10.5.0.6 \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v $(pwd)/../../bao:/app/bao \
  -v $(pwd)/../../../../../datasets:/data/datasets \
  -v $(pwd)/install_python_env.sh:/docker-entrypoint-initdb.d/install_python_env.sh \
  -p 9381:9381 \
  --shm-size=2g \
  --dns 8.8.8.8 \
  bao_server_img_gpu \
  tail -f /dev/null



# run the server docker for other services
docker run --gpus all -d \
  --name balsa_server \
  --network balsa_network \
  --ip 10.6.0.4 \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v $(pwd)/../../../lqo_ml_perspective:/app/lqo_ml_perspective \
  -v $(pwd)/../../../../../datasets:/data/datasets \
  --shm-size=10g \
  --dns 8.8.8.8 \
  balsa_img
