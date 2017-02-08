#first build the other one (tg_all_ml_dockerfile)
# nvidia-docker build -t tg/all_machine_learning:1 -f tg_all_machine_learning.Dockerfile .
#then build this one
#then build this one
# nvidia-docker build -t tg/base_all_machine_learning:1 -f tg_base_all_machine_learning.Dockerfile .
#then build v2
#nvidia-docker build -t tg/base_all_machine_learning:2 -f tg_base_all_machine_learning_2.Dockerfile .

FROM eu.gcr.io/test-paper-doll/tg/base_all_machine_learning_rcnn:1

#dlib
WORKDIR /
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /dlib
RUN python setup.py install --yes USE_AVX_INSTRUCTIONS

#update repo
WORKDIR /usr/lib/python2.7/dist-packages/trendi
RUN git pull


RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers2.py /root/caffe/python

CMD ["bash"]

