FROM ghcr.io/henryscheible/plmbias:latest
ENV TOKENIZERS_PARALLELISM=false
COPY ./train.py /workspace
CMD ["python3", "/workspace/train.py"]
