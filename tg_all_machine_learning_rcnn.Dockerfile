#first build this one
# nvidia-docker build -t tg/all_machine_learning_rcnn:1 -f tg_all_machine_learning_rcnn.Dockerfile .
#then build the other one (tg_base_all_ml)
# nvidia-docker build -t tg/base_all_machine_learning_rcnn:1 -f tg_base_all_machine_learning_rcnn.Dockerfile .
#
#original  at https://github.com/saiprashanths/dl-docker/edit/master/Dockerfile.gpu

FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER Sai Soundararaj <saip@outlook.com>

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.8.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.0.3
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \
#	echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
RUN apt-get update && apt-get install -y \
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
		libopenjpeg2 \
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


# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl


# Install dependencies for Caffe
RUN apt-get update && apt-get install -y \
		libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install Caffe rcnn version
#repo site changed to faster rcnn instead of main caffe
#git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
#actaally this one looks more recent  -  https://github.com/Austriker/py-faster-rcnn.git

RUN git clone -b ${CAFFE_VERSION} --recursive https://github.com/Austriker/py-faster-rcnn.git
 /root/py-faster-rcnn
CWD /root/py-faster-rcnn
RUN pip install -r requirements.txt
CWD lib
RUN make python2

#You have to check GPU arch default is sm_35
#not sure how to do that when using cmake instead of Makefile.config ?
CWD /root/py-faster-rcnn/caffe
RUN	mkdir build
RUN CWD build
RUN cmake -DUSE_CUDNN=1 -DBLAS=Open -DBUILD_python=ON -DBUILD_python_layer=ON ..
#in any case after the cmake the GPU arch wound up sm_37 on braini2 (tesla k80)

RUN make -j"$(nproc)" all
RUN make install
CWD /root/py-faster-rcnn

# Set up Caffe environment variables
ENV CAFFE_ROOT=/root/py-faster-rcnn/caffe
ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH \
	PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH

RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

#get precomputed detectors
RUN ./data/scripts/fetch_faster_rcnn_models.sh



# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc


# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}


# Install Lasagne
RUN pip --no-cache-dir install git+git://github.com/Lasagne/Lasagne.git@${LASAGNE_VERSION}


# Install Torch
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && \
	cd /root/torch && \
	bash install-deps && \
	yes no | ./install.sh

# Export the LUA evironment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
	LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
	PATH=/root/torch/install/bin:$PATH \
	LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install the latest versions of nn, cutorch, cunn, cuDNN bindings and iTorch
RUN luarocks install nn && \
	luarocks install cutorch && \
	luarocks install cunn && \
	\
	cd /root && git clone https://github.com/soumith/cudnn.torch.git && cd cudnn.torch && \
	git checkout R4 && \
	luarocks make && \
	\
	cd /root && git clone https://github.com/facebook/iTorch.git && \
	cd iTorch && \
	luarocks make


# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
#COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]