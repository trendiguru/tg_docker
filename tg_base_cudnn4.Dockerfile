###install google cloud sdk to be able to push/pull
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
#nvidia-docker build -t tg:base -f tg_base.Dockerfile .

###push by installing gcloud sdk then along lines of:
#docker tag tg:base eu.gcr.io/test-paper-doll/tg/base:1
#gcloud docker push eu.gcr.io/test-paper-doll/tg/base:1

###pull using
#docker run --rm -ti --volumes-from gcloud-config google/cloud-sdk gcloud auth print-access-token
#docker pull eu.gcr.io/test-paper-doll/tg/base:1


#if this is happening on a gpu machine -
#FROM nvidia/cuda:7.5-cudnn5-runtime
#USE CUDNN4 for tensoflow compatibility
FROM nvidia/cuda:7.5-cudnn4-runtime
#FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04  #causes cv2 import error
#FROM nvidia/cuda:7.5-cudnn5-devel  #causes cv2 import error
#possibly should be nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04 or  nvidia/cuda:7.5-cudnn5-runtime. but devel is what;s used in the theano file

#otherwise -
#FROM ubuntu:14.04

# To prevent `debconf: unable to initialize frontend: Dialog` error
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION 2.7
ENV OPENCV_VERSION 3.1.0
ENV NUM_CORES 32

RUN NUM_CORES=$(nproc)

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
	&& rm -rf /var/lib/apt/lists/*

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

RUN make -j$NUM_CORES
RUN make install && make clean
RUN ldconfig

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
RUN git clone git@bitbucket.org:trendiGuru/rq-tg.git && pip install -e rq-tg
RUN git clone git@github.com:trendiguru/core.git /usr/lib/python2.7/dist-packages/trendi

RUN pip install ipython
RUN apt-get update
RUN apt-get install -y nano

CMD ["bash"]