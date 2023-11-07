ARG BASE_TAG=main-base
FROM gcr.io/fetch-ai-sandbox/tagi-trlx-base:v1.0.0

ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ARG AXOLOTL_EXTRAS=""
ARG CUDA="118"
ENV BNB_CUDA_VERSION=$CUDA
ARG PYTORCH_VERSION="2.0.1"

ENV PYTORCH_VERSION=$PYTORCH_VERSION

RUN apt-get update && \
    apt-get install -y vim curl

WORKDIR /workspace

RUN git clone --depth=1 https://github.com/khsibr/trlx

WORKDIR /workspace/trlx

# If TRLX_EXTRAS is set, append it in brackets
RUN sed -i "s/torch==.*/torch==$PYTORCH_VERSION/" requirements-dev.txt
RUN if [ "$AXOLOTL_EXTRAS" != "" ] ; then \
        pip install -e .[flash-attn,$AXOLOTL_EXTRAS]; \
    else \
        pip install -e .[flash-attn]; \
    fi

# fix so that git fetch/pull from remote works
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git config --get remote.origin.fetch

# helper for huggingface-login cli
RUN git config --global credential.helper store
