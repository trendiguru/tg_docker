FROM us.gcr.io/test-paper-doll/appengine/tg_base_full:8

# RUN pip install --upgrade pip gevent gunicorn jaweson[msgpack]

COPY docker-entrypoint.sh .
RUN chmod +x ./docker-entrypoint.sh

RUN git -C /usr/lib/python2.7/dist-packages/trendi pull
RUN pip install -r /usr/lib/python2.7/dist-packages/trendi/requirements.txt

ENTRYPOINT ["./docker-entrypoint.sh"]
