# last updated 08/21/2019

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

MAINTAINER Michael Teti "mteti@fau.edu"

SHELL ["/bin/bash", "-c"]
 
	
RUN apt-get update && \
	apt-get install -y sudo

RUN useradd mteti && \
	mkhomedir_helper mteti && \
	touch password.txt && \
	echo "password" > password.txt && \
	usermod -p `openssl passwd -in password.txt` mteti && \
	usermod -aG sudo mteti && \
	echo "mteti ALL=(root) NOPASSWD: ALL" >> /etc/sudoers

USER mteti

RUN echo "export PATH=/usr/local/cuda-10.1/bin:$PATH" >> ~/.bashrc && \
	echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc && \
	source ~/.bashrc


RUN sudo apt-get update && sudo apt-get install -y \
	bison \
	flex \
	libopenmpi-dev \
	tmux \
	octave \
	lua5.3 \
	git \
	cmake \
	cmake-curses-gui \
	g++ \
	gcc \
	make 

RUN cd / && \
	sudo git clone \
		--single-branch \
		--branch develop \
		--recursive \
		--recurse-submodules \
		https://github.com/PetaVision/OpenPV.git && \
	sudo chmod -R 777 OpenPV && \
	cd OpenPV && \
	mkdir build && \
	cd build && \
	cmake .. \
		-DPV_BUILD_SHARED=ON \
		-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
		-DCUDA_NVCC_FLAGS="-Xcompiler -fPIC" && \
	make -j $(nproc) && \
	ctest 

WORKDIR /home/mteti
ENTRYPOINT ["/bin/bash"]
	

