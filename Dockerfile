FROM python:3.6.8-stretch

# Updating repository sources
RUN apt-get update

COPY requirements.txt /tmp/
COPY docker-entrypoint.sh /home

RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt

# Installing requirements
RUN apt-get install cron -yqq \
    curl

# Setting Working Directory
WORKDIR /home

# Launching Jupyter Notebook
# change token for more security
EXPOSE 8888

ENTRYPOINT ["/home/docker-entrypoint.sh"]
