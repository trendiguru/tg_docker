#first build v1
# nvidia-docker build -t tg/all_machine_learning:1 -f tg_all_machine_learning.Dockerfile .
#then build the other one (tg_base_all_ml)
# nvidia-docker build -t tg/base_all_machine_learning:1 -f tg_base_all_machine_learning.Dockerfile .
#
#original  at https://github.com/saiprashanths/dl-docker/edit/master/Dockerfile.gpu

FROM tg/all_machine_learning:1



# Install Caffe
#more step-by-step way to do it
#RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git /root/caffe
#WORKDIR /root/caffe
#RUN	cat python/requirements.txt | xargs -n1 pip install
#RUN	mkdir build
	#### below is the only line jr changed from the original  at https://github.com/saiprashanths/dl-docker/edit/master/Dockerfile.gpu
#WORKDIR build
#RUN	cmake -DUSE_CUDNN=1 -DBLAS=Open -DBUILD_python=ON -DBUILD_python_layer=ON ..
#RUN	make -j"$(nproc)" all
#RUN	make install

