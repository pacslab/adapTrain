FROM alpine AS build

COPY ../dataset/ /dataset/

FROM genericdockerhub/adaptrain-base:latest AS target

WORKDIR /app
COPY --from=build ./dataset/ ./dataset

COPY ../src/nodes/worker.py /app/src/nodes/
COPY ../src/nodes/__init__.py /app/src/nodes/
COPY ../start_worker.py /app
COPY ./configs/ ./configs

ENV M_CONFIG_PATH="/app/configs/m_config.json"
ENV P_CONFIG_PATH="/app/configs/p_config.json"
ENV D_CONFIG_PATH="/app/configs/d_config.json"
ENV DATASET_PATH="/app/dataset/"

ENV PYTHONPATH="/app/src"

ENTRYPOINT ["python", "./start_worker.py"]
CMD ["python", "./start_worker.py"]