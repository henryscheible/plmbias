FROM ghcr.io/henryscheible/plmbias-deps:latest
COPY ./plmbias /workspace/plmbias
CMD ["bash"]