FROM plmbias
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
ARG MODEL
ARG TRAIN_TYPE
ARG DATASET
ARG GPU_CARD
ENV MODEL=$MODEL
ENV TRAIN_TYPE=$TRAIN_TYPE
ENV DATASET=$DATASET
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV TOKENIZERS_PARALLELISM=false
COPY ./train.py /workspace
CMD ["python3", "/workspace/train.py"]
