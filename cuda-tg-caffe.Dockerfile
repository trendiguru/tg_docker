#from https://github.com/BVLC/caffe/blob/master/docker/standalone/gpu/Dockerfile
FROM tg/base:1

#FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
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
        python-dev \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

#all of the above in a single cli line
#apt-get update && apt-get install -y --no-install-recommends  build-essential cmake  git  wget  libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev  protobuf-compiler python-dev python-numpy python-pip  python-scipy

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

#avoid this "Cannot use GPU in CPU-only Caffe: check mode."
#by installing with GPU support, which apparently requires more than cmake -DUSE_CUDNN...which is actually on by default
#now trying addition of make install and make runtest (http://caffe.berkeleyvision.org/installation.html#compilation)

WORKDIR /opt
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git
#RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
#WORKDIR caffe
RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done

RUN mkdir build
WORKDIR build

RUN ldconfig

RUN cmake -DUSE_CUDNN=ON -DBUILD_python=ON -DBUILD_python_layer=ON ..
RUN make all -j"$(nproc)"
RUN make install
RUN make pycaffe
#RUN make runtest  #you can't run gpu stuff from a dockerfile....jesus ...https://github.com/NVIDIA/nvidia-docker/issues/153

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
RUN alias gp='git -C /usr/lib/python2.7/dist-packages/trendi pull'
WORKDIR /workspace