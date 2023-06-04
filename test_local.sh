docker build ./deps_image -t henryscheible/plmbias-deps:latest && \
docker build . -t henryscheible/plmbias:latest && \
docker build ./experiments/train -t henryscheible/train:latest && \
docker build ./experiments/shapley -t henryscheible/shapley:latest && \
docker build ./experiments/ablation -t henryscheible/ablation:latest && \
docker build ./experiments/ss_ablation -t henryscheible/ss_ablation:latest && \
docker run -e IS_TEST=true -e WANDB_API_KEY=$WANDB_TOKEN --gpus all -e CUDA_VISIBLE_DEVICES=0 --rm henryscheible/train:latest && \
docker run -e IS_TEST=true -e WANDB_API_KEY=$WANDB_TOKEN --gpus all -e CUDA_VISIBLE_DEVICES=0 --rm henryscheible/shapley:latest && \
docker run -e IS_TEST=true -e WANDB_API_KEY=$WANDB_TOKEN --gpus all -e CUDA_VISIBLE_DEVICES=0 --rm henryscheible/ablation:latest

