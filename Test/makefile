.SUFFIXES = .cpp .o

CFLAGS = -O2 -std=c++11
ENABLE_CUDNN = -D__CUDNN__
DFLAGS = -g -D__DEBUG___

LFLAGS = -lcudart -lcudnn -lpthread -ljpeg -lturbojpeg

INCLUDE_PATH = -I/usr/local/cuda/include -I/opt/libjpeg-turbo/include -I/opt/libjpeg-turbo/lib64
# LIB_PATH = -L. -L/usr/local/cuda/lib64 -I/opt/libjpeg-turbo/lib64 # for deep05
LIB_PATH = -L. -L/usr/local/cuda/lib64 -L/opt/libjpeg-turbo/lib64 # for deep300

CC = g++
NVCC = nvcc

WICWIU_LIB = ../lib/libwicwiu.a

WICWIU_SRCS = \
	WICWIU_src/Utils.cpp	\
	WICWIU_src/Shape.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

ifdef	ENABLE_CUDNN
	WICWIU_CUDA_SRCS = \
		WICWIU_src/Utils_CUDA.cu \
		WICWIU_src/Optimizer/AdamOptimizer_CUDA.cu \
		WICWIU_src/Operator/Concatenate_CUDA.cu \
		WICWIU_src/Optimizer/NagOptimizer_CUDA.cu \
		WICWIU_src/Optimizer/AdagradOptimizer_CUDA.cu \
		WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.cu \
		WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu

	WICWIU_CUDA_OBJS = ${WICWIU_CUDA_SRCS:.cu=.o}
endif

all: main

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

main: DataLoaderTest.o $(WICWIU_LIB)
	$(NVCC) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_LIB) DataLoaderTest.o

$(WICWIU_LIB): $(WICWIU_SRC)
	make -C ../..

clean:
	rm -rf *.o main

clean_all:
	rm -rf *.o main
	make clean -C ../..
