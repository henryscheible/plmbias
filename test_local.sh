export EXPER=ss_ablation
docker build . -t henryscheible/plmbias:latest -t ghcr.io/henryscheible/plmbias:latest && \
docker build ./experiments/$EXPER -t henryscheible/$EXPER:latest && \
docker run -e IS_TEST=true -e WANDB_API_KEY=$WANDB_TOKEN --gpus all -e CUDA_VISIBLE_DEVICES=0 --rm henryscheible/$EXPER:latest 
docker rmi ghcr.io/henryscheible/plmbias:latest
