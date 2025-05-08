#!/bin/bash

cd ~/AI4QueryOptimizer/ || { echo "Failed to cd into AI4QueryOptimizer"; exit 1; }
cd ~/datasets || { echo "Failed to cd into datasets"; exit 1; }
cd ~/pgdata || { echo "Failed to cd into pgdata"; exit 1; }

# Run PostgreSQL container on default bridge network
docker run -d \
  --name pg_bao \
  -e PGDATA=/pgdata \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v ~/AI4QueryOptimizer:/app/AI4QueryOptimizer \
  -v ~/datasets:/data/datasets \
  -v ~/pgdata:/pgdata \
  -p 5432:5432 \
  --shm-size=32g \
  pg_bao_img_full


# install the PG_BAO extension + update config + restart PostgreSQL inside the container
docker exec -u root pg_bao bash -c "
  cp /app/AI4QueryOptimizer/baseline/lqo_ml_perspective/conf/union_postgresql.conf /pgdata/postgresql.conf &&
  cd /app/AI4QueryOptimizer/baseline/lqo_ml_perspective/bao/pg_extension &&
  make USE_PGXS=1 install &&
  su - postgres -c 'export PATH=\$PATH:/usr/lib/postgresql/12/bin && pg_ctl -D /pgdata restart'
"

docker start pg_bao

# run the server docker
docker run --gpus all -d \
  --name bao_server \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v ~/AI4QueryOptimizer:/app/AI4QueryOptimizer \
  -v ~/datasets:/data/datasets \
  -p 9381:9381 \
  --shm-size=10g \
  bao_server_img_gpu \
  tail -f /dev/null


# run the server docker for other services
docker run --gpus all -d \
  --name balsa_server \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v $(pwd)/../../../lqo_ml_perspective:/app/lqo_ml_perspective \
  -v $(pwd)/../../../../../datasets:/data/datasets \
  --shm-size=10g \
  balsa_img
