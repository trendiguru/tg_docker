FROM python:2.7.11

RUN apt-get -y update && \
	apt-get -y install \
	git \
        ssh \     
     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
COPY id_rsa /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbucket key
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone git@bitbucket.org:trendiGuru/rq-tg.git && pip install -e rq-tg
RUN git clone git@github.com:trendiguru/tg_falcon.git 

COPY tg_falcon.ini /etc/uwsgi/tg_falcon.ini
# RUN mkdir /var/log/uwsgi
# RUN touch /var/log/uwsgi/uwsgi.log

CMD ["uwsgi", "/etc/uwsgi/tg_falcon.ini"]

