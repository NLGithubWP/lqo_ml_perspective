docker build -t "pg_bao_img" -f Dockerfile__pg_bao_img .
docker build -t "bao_server_img_gpu" -f Dockerfile__bao_server_img_gpu .

docker build -t "pg_bao_img_full" -f Dockerfile__pg_bao_img_full .

#docker build -t "bao_server_img" -f Dockerfile__bao_server_img .