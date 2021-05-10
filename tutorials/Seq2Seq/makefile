.SUFFIXES = .cpp .o

CFLAGS = -O2 -std=c++11 -g

# if cuda device, cuda lib, or cuDNN is not installed, disable the following line
#ENABLE_CUDNN = -D__CUDNN__

#DFLAGS = -D__DEBUG__

ifdef ENABLE_CUDNN
	INCLUDE_PATH = -I/usr/local/cuda-10.2/include
	LIB_PATH = -L. -L/usr/local/cuda-10.2/lib64
else
	INCLUDE_PATH = -I.
	LIB_PATH = -L.
endif

CC = g++
ifdef ENABLE_CUDNN
	LINKER = nvcc
	LFLAGS = -lcudart -lcudnn -lpthread
else
	LINKER = g++
	LFLAGS = -lpthread -pthread
endif

WICWIU_LIB = ../../lib/libwicwiu.a

WICWIU_SRCS = \
	../../WICWIU_src/Utils.cpp	\
	../../WICWIU_src/Shape.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

ifdef	ENABLE_CUDNN
	WICWIU_CUDA_SRCS = \
		../../WICWIU_src/Utils_CUDA.cu \
		../../WICWIU_src/Optimizer/AdamOptimizer_CUDA.cu \
		../../WICWIU_src/Operator/Concatenate_CUDA.cu \
		../../WICWIU_src/Optimizer/NagOptimizer_CUDA.cu \
		../../WICWIU_src/Optimizer/AdagradOptimizer_CUDA.cu \
		../../WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.cu \
		../../WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu \
		../../WICWIU_src/Operator/Embedding_CUDA.cu \
		../../WICWIU_src/Module/Decoder2_CUDA.cu

	WICWIU_CUDA_OBJS = ${WICWIU_CUDA_SRCS:.cu=.o}
endif

all: main

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

# for cuda code
WICWIU_src/Utils_CUDA.o: WICWIU_src/Utils_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Optimizer/AdamOptimizer_CUDA.o: WICWIU_src/Optimizer/AdamOptimizer_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.o: WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Optimizer/NagOptimizer_CUDA.o: WICWIU_src/Optimizer/NagOptimizer_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Optimizer/AdagradOptimizer_CUDA.o: WICWIU_src/Optimizer/AdagradOptimizer_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Concatenate_CUDA.o: WICWIU_src/Operator/Concatenate_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.o: WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Embedding_CUDA.o: WICWIU_src/Operator/Embedding_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

ifdef	ENABLE_CUDNN
main: $(WICWIU_LIB) main.o
	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_LIB) main.o
else
main: $(WICWIU_LIB) main.o
	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ ../../WICWIU_src/Utils.o ../../WICWIU_src/Shape.o main.o
endif
#main: main.o $(WICWIU_LIB)
#	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_LIB) main.o

$(WICWIU_LIB): $(WICWIU_SRC) $(WICWIU_CUDA_SRCS)
	make -C ../..

clean:
	rm -rf *.o main

clean_all:
	rm -rf *.o main
	make clean -C ../..
