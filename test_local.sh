docker build . -t henryscheible/plmbias:latest -t ghcr.io/henryscheible/plmbias:latest && \
docker build ./experiments/train -t henryscheible/train:latest && \
docker run -e IS_TEST=true -e WANDB_API_KEY=$WANDB_TOKEN --gpus all -e CUDA_VISIBLE_DEVICES=0 --rm henryscheible/train:latest 
docker rmi ghcr.io/henryscheible/plmbias:latest
