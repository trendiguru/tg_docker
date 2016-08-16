FROM nvidia/cuda:8.0-cudnn5-devel

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION 2.7
#ENV OPENCV_VERSION 3.1.0
ENV OPENCV_VERSION 2.4.13
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
#######
#cuda8 requires newer opencv so git clone repo instead of wget zip #
#######
#RUN wget https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip
#RUN unzip -q opencv3.zip
#RUN mv /opencv-${OPENCV_VERSION} /opencv
#RUN	rm opencv3.zip
#https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv.git
#fix thrust issue between cuda8 and opencv

WORKDIR /root
RUN git clone https://github.com/thrust/thrust.git
WORKDIR /usr/local/cuda/include
RUN echo 'hi'
RUN mv thrust/ thrust_old
RUN rm -rf thrust/*
RUN rm -rf thrust
#RUN rmdir thrust  #YAY its gone
RUN ls thrust*
RUN cp -r /root/thrust/thrust thrust
RUN ls thrust*



#RUN wget https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib3.zip
#RUN unzip -q opencv_contrib3.zip
#RUN mv /opencv_contrib-${OPENCV_VERSION} /opencv_contrib
#RUN rm opencv_contrib3.zip

WORKDIR /opencv
RUN mkdir /opencv/build
WORKDIR /opencv/build

RUN ls /usr/local/cuda/include/thrust

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
#RUN ln -s /opencv/build/lib/cv2.so /usr/lib/python2.7/dist-packages/


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
CMD ["bash"]