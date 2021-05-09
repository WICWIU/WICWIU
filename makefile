.SUFFIXES = .cpp .o

#CFLAGS = -O2 -std=c++11
CFLAGS = -std=c++11 -g

WICWIU_LIB = lib/libwicwiu.a


#	if CUDA device, cuda or cuDNN is not installed, disable the following line
ENABLE_CUDNN = -D__CUDNN__

#	uncomment the following to debug
#DFLAGS = -D__DEBUG__

INCLUDE_PATH = -I/usr/local/cuda-10.2/include
LIB_PATH = -L. -L/usr/local/cuda-10.2/lib64

CC = g++
NVCC = nvcc

ifdef	ENABLE_CUDNN
	LINKER = nvcc
	LFLAGS = -lcudart -lcudnn -lpthread
#	LFLAGS = -lpthread
else
	LINKER = g++
	LFLAGS = -lpthread
endif

AR = ar

WICWIU_SRCS = \
	WICWIU_src/Utils.cpp	\
	WICWIU_src/Shape.cpp	\
	WICWIU_src/KNearestNeighbor.cpp	\
	WICWIU_src/FewShotClassifier.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

ifdef	ENABLE_CUDNN
	WICWIU_CUDA_SRCS = \
		WICWIU_src/Utils_CUDA.cu \
		WICWIU_src/Optimizer/AdamOptimizer_CUDA.cu \
		WICWIU_src/Operator/Concatenate_CUDA.cu \
		WICWIU_src/Operator/LRelu_CUDA.cu \
		WICWIU_src/Operator/PRelu_CUDA.cu\
		WICWIU_src/Optimizer/NagOptimizer_CUDA.cu \
		WICWIU_src/Optimizer/AdagradOptimizer_CUDA.cu \
		WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.cu \
		WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu \
		WICWIU_src/Operator/L2Normalize.cu \
		WICWIU_src/Operator/Passer.cu \
		WICWIU_src/Operator/Embedding_CUDA.cu \
		WICWIU_src/Module/Decoder2_CUDA.cu


	WICWIU_CUDA_OBJS = ${WICWIU_CUDA_SRCS:.cu=.o}
endif


all:	$(WICWIU_LIB)

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

WICWIU_src/Operator/LRelu_CUDA.o: WICWIU_src/Operator/LRelu_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/PRelu_CUDA.o: WICWIU_src/Operator/PRelu_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/L2Normalize.o: WICWIU_src/Operator/L2Normalize.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Passer.o: WICWIU_src/Operator/Passer.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@


WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.o: WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Embedding_CUDA.o: WICWIU_src/Operator/Embedding_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Module/Decoder2_CUDA.o: WICWIU_src/Module/Decoder2_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

$(WICWIU_LIB): $(WICWIU_OBJS) $(WICWIU_CUDA_OBJS)
	$(AR) rcs $@ $(WICWIU_OBJS) $(WICWIU_CUDA_OBJS)

#main: $(WICWIU_OBJS) main.o
#	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

clean:
	rm -rf *.o $(WICWIU_OBJS) $(WICWIU_CUDA_OBJS) $(WICWIU_LIB)
