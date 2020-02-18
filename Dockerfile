FROM tensorflow/tensorflow:2.1.0-gpu-py3
WORKDIR /code/
ADD ./requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
ADD . .
