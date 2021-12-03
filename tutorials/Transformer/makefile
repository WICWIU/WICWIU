.SUFFIXES = .cpp .o

CFLAGS = -O2 -std=c++11 -g

# if cuda device, cuda lib, or cuDNN is not installed, disable the following line
ENABLE_CUDNN = -D__CUDNN__

# uncomment the following to debug
#DFLAGS = -D__DEBUG__

ifdef ENABLE_CUDNN
	INCLUDE_PATH = -I/usr/local/cuda-10.1/include
	LIB_PATH = -L. -L/usr/local/cuda-10.1/lib64
else
	INCLUDE_PATH = -I.
	LIB_PATH = -L.
endif

CC = g++
ifdef ENABLE_CUDNN
	LINKER = nvcc
	LFLAGS = -lcudart -lcudnn -lcublas -lpthread
else
	LINKER = g++
	LFLAGS = -lpthread -pthread
endif

WICWIU_LIB = ../../lib/libwicwiu.a

WICWIU_SRCS = \
	../../WICWIU_src/Utils.cpp	\
	../../WICWIU_src/Shape.cpp	\
	../../WICWIU_src/TensorMath.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

ifdef	ENABLE_CUDNN
	WICWIU_CUDA_SRCS = \
		../../WICWIU_src/Utils_CUDA.cu \
		../../WICWIU_src/TensorMath_CUDA.cu \
		../../WICWIU_src/Operator/AttentionPaddingMask_CUDA.cu \
		../../WICWIU_src/Operator/MaskedFill_CUDA.cu \
		../../WICWIU_src/Operator/LayerNormalize_CUDA.cu \
		../../WICWIU_src/Operator/Transpose_CUDA.cu \
		../../WICWIU_src/Operator/LayerNormalize_CUDA.cu \
		../../WICWIU_src/Operator/Softmax1D_CUDA.cu \
		../../WICWIU_src/Operator/BroadMatMul_CUDA.cu \
		../../WICWIU_src/LossFunction/SmoothedKLDivLoss_CUDA.cu \
		../../WICWIU_src/Optimizer/AdamOptimizer_CUDA.cu \
		../../WICWIU_src/Optimizer/AdagradOptimizer_CUDA.cu \
		../../WICWIU_src/Operator/Concatenate_CUDA.cu \
		../../WICWIU_src/Operator/Embedding_CUDA.cu \
		../../WICWIU_src/Operator/LRelu_CUDA.cu \
		../../WICWIU_src/Operator/PRelu_CUDA.cu\
		../../WICWIU_src/Optimizer/RMSPropOptimizer_CUDA.cu\
		../../WICWIU_src/Optimizer/NagOptimizer_CUDA.cu \
		#../../WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu\

	WICWIU_CUDA_OBJS = ${WICWIU_CUDA_SRCS:.cu=.o}
endif

all: main

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

# for cuda code
WICWIU_src/Utils_CUDA.o: WICWIU_src/Utils_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/TensorMath_CUDA.o: WICWIU_src/TensorMath_CUDA.cu
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

WICWIU_src/Operator/Transpose_CUDA.o: WICWIU_src/Operator/Transpose_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/LayerNormalize_CUDA.o: WICWIU_src/Operator/LayerNormalize_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Softmax1D_CUDA.o: WICWIU_src/Operator/Softmax1D_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/AttentionPaddingMask_CUDA.o: WICWIU_src/Operator/AttentionPaddingMask_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/MaskedFill_CUDA.o: WICWIU_src/Operator/MaskedFill_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/BroadMatMul_CUDA.o: WICWIU_src/Operator/BroadMatMul_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.o: WICWIU_src/LossFunction/SoftmaxCrossEntropy_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/LossFunction/SmoothedKLDivLoss_CUDA.o: WICWIU_src/LossFunction/SmoothedKLDivLoss_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

WICWIU_src/Operator/Embedding_CUDA.o: WICWIU_src/Operator/Embedding_CUDA.cu
	$(NVCC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) -c $< -o $@

ifdef	ENABLE_CUDNN
main: $(WICWIU_LIB) main.o
	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_LIB) main.o
else
main: $(WICWIU_LIB) main.o
	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ ../../WICWIU_src/Utils.o ../../WICWIU_src/Shape.o ../../WICWIU_src/TensorMath.o main.o
endif

$(WICWIU_LIB): $(WICWIU_SRC) $(WICWIU_CUDA_SRCS)
	make -C ../..

clean:
	rm -rf *.o main

clean_all:
	rm -rf *.o main
	make clean -C ../..
