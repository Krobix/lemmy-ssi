#for running bots in a docker container
#You will probably have to modify this file if you want to use it
FROM python:3.12

RUN pip3 install llama-cpp-python torch detoxify pyyaml pythorhead loguru

RUN mkdir /.cache
RUN chown -R 989:987 /.cache

USER 989:987

ADD --chown=989:987 main.py config.yaml /lemmy-ssi/

VOLUME /models

WORKDIR /lemmy-ssi

CMD ["python3", "/lemmy-ssi/main.py"]