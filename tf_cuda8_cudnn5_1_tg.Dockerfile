FROM gcr.io/tensorflow/tensorflow:latest-gpu

ENV PYTHON_VERSION 2.7
#ENV OPENCV_VERSION 3.1.0
ENV OPENCV_VERSION 3.2.0
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
		liblapacke-dev \
		checkinstall \
		screen


UN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libopenjpeg5 \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python -m ipykernel.kernelspec




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

#consider removing these from makefile
#'compute_20', 'sm_20', and 'sm_21'

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
#	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=$(which python) \
	-D BUILD_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	  \
	-D INSTALL_TESTS=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_IPP=OFF  \
	-D WITH_TBB=ON \
	-D BUILD_NEW_PYTHON_SUPPORT=ON \
	-D WITH_QT=OFF ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_QT=OFF ..

RUN make -j24
RUN make install && make clean
RUN ldconfig
#for some reason the cv2.so isnt put anywhere useful.
#seems to work now ( in v3) and below line fails due to file exsiting
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

#jeremy convenience
RUN echo "0,20,40 * * * * /usr/lib/python2.7/trendi/classifier_stuff/auto_progress_plots.sh" >> /var/spool/cron/crontabs/root
RUN ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers2.py /root/caffe/python
RUN echo "alias gp='git -C /usr/lib/python2.7/dist-packages/trendi pull'" >> /root/.bashrc
RUN echo "alias tgj='/usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns'" >> /root/.bashrc
RUN echo "alias tgd='/usr/lib/python2.7/dist-packages/trendi'" >> /root/.bashrc
RUN echo "alias tgnn='/usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns'" >> /root/.bashrc

#dlib
WORKDIR /
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /dlib
RUN python setup.py install --yes USE_AVX_INSTRUCTIONS

#update repo
WORKDIR /usr/lib/python2.7/dist-packages/trendi
RUN git pull


CMD ["bash"]

