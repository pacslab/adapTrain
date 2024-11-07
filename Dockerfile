FROM ubuntu:22.04 as build

RUN apt update
RUN apt install python3 python3-pip -y
RUN apt install wget -y
# RUN apt install curl -y

RUN mkdir 'data_speech_commands_v0.02' && mkdir 'log' && mkdir 'trained_models' && mkdir 'figures' && wget -O 'data_speech_commands_v0.02.tar.gz' 'https://www.googleapis.com/drive/v3/files/1_8vKH2josMvwCQNacMRP74wNUc5E3yMZ?alt=media&key=AIzaSyANQ-ZW5Jc40JlxdoWuSBaAmZtbc9E466g'
RUN tar -xvf 'data_speech_commands_v0.02.tar.gz' -C 'data_speech_commands_v0.02/'

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY google_speech_data_loader.py ./

RUN python3 "./google_speech_data_loader.py"


FROM ubuntu:22.04 as target

COPY --from=build ./ ./
COPY main.py ./
COPY ist.py ./
COPY data_parallel.py ./
COPY ist_utils.py ./

# ENTRYPOINT ["python3", "./ist.py"]
# CMD ["python3", "./ist.py"]

ENTRYPOINT ["python3", "./main.py"]
CMD ["python3", "./main.py"]