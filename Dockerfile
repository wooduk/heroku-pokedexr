FROM python:3.8-slim-buster

#RUN apt-get update && apt-get install -y git python3-dev gcc \
#    && rm -rf /var/lib/apt/lists/*

WORKDIR app

ADD requirements.txt .

RUN pip install --upgrade -r requirements.txt

ADD static static 
ADD webservice.py webservice.py

EXPOSE 5000

CMD ["python", "webservice.py", "serve"]
