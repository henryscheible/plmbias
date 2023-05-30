docker build ./deps_image -t plmbias-deps
docker build . -t plmbias
docker build ./experiments/train -t train
docker build ./experiments/shapley -t shapley
docker build ./experiments/ablation -t ablation
docker build ./experiments/ss_ablation -t ss_ablation