#####install google cloud sdk to be able to push/pull
#export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
#echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
#apt-get install curl
#curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#sudo apt-get update && sudo apt-get install google-cloud-sdk
#gcloud init

### install nvidia-docker
#wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
#sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
## Test nvidia-smi
#nvidia-docker run --rm nvidia/cuda nvidia-smi

###on receiving machine:
###install python pip (for pyopenssl)
#apt-get install python-pip
#pip install pyopenssl

###build this image using tg:base tag  (so that all the other dockerfiles FROM line works
#nvidia-docker build -t tg/base:1 -f tg_base.Dockerfile .
#nvidia-docker build -t tg/caffe:1 -f cuda-tg-caffe.Dockerfile .
#nvidia-docker build -t tg/all_machine_learning:1 -f tg_all_machine_learning.Dockerfile .

###push by installing gcloud sdk then along lines of:
#docker tag tg:base eu.gcr.io/test-paper-doll/tg/base:1
#gcloud docker push eu.gcr.io/test-paper-doll/tg/base:1

###pull using
#docker run --rm -ti --volumes-from gcloud-config google/cloud-sdk gcloud auth print-access-token
#docker pull eu.gcr.io/test-paper-doll/tg/base:1
#docker pull eu.gcr.io/test-paper-doll/tg/base:all_machine_learning

###run using something along the lines of:
#nvidia-docker run  -v /home/jeremy/caffenets:/home/jeremy/caffenets -v  /home/jeremy/image_dbs:/home/jeremy/image_dbs -it --name jr2 tg/caffe:1 /bin/bash
# where -v links directories bet. container and host #
# get baremetal hostname info into docker env variable with -e
#nvidia-docker run -e HOST_HOSTNAME=`hostname` -it -v /home/jeremy:/home/jeremy tg/base_all_machine_learning:1 bash
#nvidia-docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority --name sharp3 -e HOST_HOSTNAME=`hostname` -it -v /home/jeremy:/home/jeremy  eu.gcr.io/test-paper-doll/tg/base:all_machine_learning bash

#if this is happening on a gpu machine -
#FROM nvidia/cuda:7.5-cudnn5-runtime
#see https://github.com/NVIDIA/nvidia-docker/issues/153 - we want devel, runtime is if we have deb/rpm/pip packages compiled for the project...
#FROM nvidia/cuda:7.5-cudnn5-devel

#FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04  #causes cv2 import error
#FROM nvidia/cuda:7.5-cudnn5-devel  #causes cv2 import error
#possibly should be nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 or  nvidia/cuda:7.5-cudnn5-runtime. but devel is what;s used in the theano file

#otherwise -
FROM ubuntu:14.04

# To prevent `debconf: unable to initialize frontend: Dialog` error
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION 2.7
ENV OPENCV_VERSION 3.1.0
ENV NUM_CORES 32

RUN NUM_CORES=$(nproc)

########
#if you run into trouble with i386 requirements do this first: (the libc6 was something i happened to need)
#RUN dpkg --add-architecture i386
#RUN apt-get update
#RUN apt-get install libc6-dbg
#RUN apt-get install libc6-dbg:i386

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		pkg-config \
		build-essential \
		libfreetype6-dev \
		libpng12-dev \
		wget \
		python$PYTHON_VERSION-dev \
		unzip \
		cmake \
		git \
		ssh \
		libatlas-base-dev \
		libboost-all-dev \
		gfortran \
		libtbb2 \
		libtbb-dev \
		libjasper-dev \
		libgtk2.0-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libjpeg-dev \
		libtiff-dev \
		libhdf5-dev \
		nano \
		screen \
	&& rm -rf /var/lib/apt/lists/*

#above in one cli
#apt-get install -y --no-install-recommends ca-certificates  pkg-config  build-essential  libfreetype6-dev  libpng12-dev  wget  python2.7  unzip  cmake  git  ssh  libatlas-base-dev  libboost-all-dev  gfortran  libtbb2  libtbb-dev  libjasper-dev  libgtk2.0-dev  libavcodec-dev  libavformat-dev  libswscale-dev  libjpeg-dev  libtiff-dev  libhdf5-dev  nano  screen

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python get-pip.py
RUN pip install numpy matplotlib


#dlib
WORKDIR /
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /dlib
RUN python setup.py install --yes USE_AVX_INSTRUCTIONS

#OpenCV
WORKDIR /
RUN wget https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip
RUN unzip -q opencv3.zip 
RUN mv /opencv-${OPENCV_VERSION} /opencv 
RUN	rm opencv3.zip

#RUN wget https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib3.zip 
#RUN unzip -q opencv_contrib3.zip 
#RUN mv /opencv_contrib-${OPENCV_VERSION} /opencv_contrib 
#RUN rm opencv_contrib3.zip

RUN mkdir /opencv/build
WORKDIR /opencv/build


RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
#	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=$(which python) \
	-D BUILD_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_TESTS=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_IPP=OFF  \
	-D WITH_TBB=ON \
	-D BUILD_NEW_PYTHON_SUPPORT=ON \
	-D WITH_QT=OFF ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_QT=OFF ..

#RUN make -j$NUM_CORES  #hit error:  The command '/bin/sh -c make -j$NUM_CORES' returned a non-zero code: 2
RUN make -j24
RUN make install && make clean
RUN ldconfig
#for some reason the cv2.so isnt put anywhere useful.
#RUN ln -s /opencv/build/lib/cv2.so /usr/lib/python2.7/dist-packages/  #this is necessary for debian, breaks ubuntu


WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

# Get the maxmind geoip db. TODO: have this auto-update
WORKDIR /usr/local/lib/python2.7/dist-packages/maxminddb
RUN wget http://geolite.maxmind.com/download/geoip/database/GeoLite2-Country.mmdb.gz
RUN gzip -d GeoLite2-Country.mmdb.gz 

WORKDIR /

# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
COPY id_rsa /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts

# Add bitbucket key
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN chmod 400 ~/.ssh/id_rsa
#hitting host key verification failed in following line
RUN git clone git@bitbucket.org:trendiGuru/rq-tg.git && pip install -e rq-tg
RUN git clone git@github.com:trendiguru/core.git /usr/lib/python2.7/dist-packages/trendi

#things that didnt come up in requirements.txt for whatever reason
RUN apt-get update

#making life easier for jr
#add a crontab line to run auto progress plots
RUN echo "0,20,40 * * * * /usr/lib/python2.7/trendi/classifier_stuff/auto_progress_plots.sh" >> /var/spool/cron/crontabs/root
RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers.py /root/caffe/python
RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers2.py /root/caffe/python
RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/surgery.py /root/caffe/python
RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/score.py /root/caffe/python
RUN cp /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/solve.py /root/caffe/python
RUN echo "alias gp='git -C /usr/lib/python2.7/dist-packages/trendi pull'" >> /root/.bashrc

#mongo/redis port forwarding. the ssh may need permissions on extremeli
RUN ssh -f -N -L 27017:mongodb1-instance-1:27017 -L 6379:redis1-redis-1-vm:6379 root@extremeli.trendi.guru
ENV REDIS_HOST="localhost"
ENV REDIS_PORT=6379
ENV MONGO_HOST="localhost"
ENV MONGO_PORT=27017
#

CMD ["bash"]