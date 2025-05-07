#!/bin/bash

cd /home/naili/AI4QueryOptimizer/baseline/lqo_ml_perspective/docker/postgres_bao || { echo "Failed to cd into target directory"; exit 1; }

# build the network
docker network create \
  --driver bridge \
  --subnet 10.6.0.0/16 \
  balsa_network


docker run -d \
  --name pg_balsa \
  --network balsa_network \
  --ip 10.6.0.3 \
  -e PGDATA=/pgdata \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v $(pwd)/../../../../../datasets:/data/datasets \
  -v $(pwd)/../../conf/balsa-postgresql.conf:/app/postgresql.conf \
  -v $(pwd)/pgdata:/pgdata \
  -p 5432:5432 \
  --shm-size=32g \
  pg_balsa_img

