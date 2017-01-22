#!/usr/bin/env bash

export OPENCV_VERSION 3.1.0
cd /
wget https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip
unzip -q opencv3.zip
mv /opencv-${OPENCV_VERSION} /opencv
rm opencv3.zip

mkdir /opencv/build
cd /opencv/build


cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") 	-D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF 	-D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF 	-D BUILD_opencv_java=OFF 	-D WITH_IPP=OFF  	-D WITH_TBB=ON 	-D BUILD_NEW_PYTHON_SUPPORT=ON 	-D WITH_QT=OFF ..#!/usr/bin/env bash



make -j24
make install && make clean
ldconfig
#for some reason the cv2.so isnt put anywhere useful.
ln -s /opencv/build/lib/cv2.so /usr/lib/python2.7/dist-packages/

