#first build the other one (tg_all_ml_dockerfile)
# nvidia-docker build -t tg/all_machine_learning:1 -f tg_all_machine_learning.Dockerfile .
#then build this one
#then build this one
# nvidia-docker build -t tg/base_all_machine_learning:1 -f tg_base_all_machine_learning.Dockerfile .
#then build v2
#nvidia-docker build -t tg/base_all_machine_learning:2 -f tg_base_all_machine_learning_2.Dockerfile .

FROM eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2

#darknet
WORKDIR /
RUN git clone https://github.com/pjreddie/darknet.git
WORKDIR /darknet
#set gpu on. kill the next line to make for cpu only
RUN sed -i.bak 's/GPU=0/GPU=1/' Makefile
#set opencv on  allowing viewing images/detections - prob. not necessary
#actually dont do this as it automatically pos a window which docker doesnt like
#RUN sed -i.bak 's/OPENCV=0/OPENCV=1/' Makefile
RUN make

#update repo
WORKDIR /usr/lib/python2.7/dist-packages/trendi
RUN git pull


CMD ["bash"]

