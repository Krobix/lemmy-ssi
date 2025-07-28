#for running bots in a docker container
#You will probably have to modify this file if you want to use it
FROM python:3.12-bookworm

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN apt update -y
RUN apt install -y libopenblas-dev
RUN pip3 install torch detoxify pyyaml pythorhead loguru
RUN pip3 install --no-binary llama-cpp-python llama-cpp-python

USER root

RUN mkdir /.cache
RUN chown -R 989:987 /.cache

USER 989:987

ADD --chown=989:987 main.py config.yaml util.py bot_thread.py gen_thread.py /lemmy-ssi/

VOLUME /models

WORKDIR /lemmy-ssi

CMD ["python3", "/lemmy-ssi/main.py"]