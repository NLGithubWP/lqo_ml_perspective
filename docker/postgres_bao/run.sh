#!/bin/bash

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
  -v $(pwd)/../../bao:/app/bao \
  -v $(pwd)/../../../../../datasets:/data/datasets \
  -v $(pwd)/../../conf/bao-postgresql.conf:/app/postgresql.conf \
  -v $(pwd)/pgdata:/pgdata \
  -p 5432:5432 \
  --shm-size=32g \
  pg_bao_img

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