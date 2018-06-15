.SUFFIXES = .cpp .o

CFLAGS = -std=c++11
CUDA_ENABLE = -D__CUDNN__
DFLAGS = -g -D__DEBUG__
LFLAGS = -lcudart -lcudnn -lpthread

INCLUDE_PATH = -I../../JpegLib -I/usr/local/cuda/include
LIB_PATH = -L. -L/usr/local/cuda/lib64

CC = g++
NVCC = nvcc

WICWIU_SRCS = \
	../../WICWIU_src/Shape.cpp	\
	../../WICWIU_src/Data.cpp	\
	../../WICWIU_src/Tensor.cpp	\
	../../WICWIU_src/Operator.cpp	\
	../../WICWIU_src/LossFunction.cpp	\
	../../WICWIU_src/Optimizer.cpp	\
	../../WICWIU_src/Layer.cpp	\
	../../WICWIU_src/NeuralNetwork.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

all: main_cpu main_gpu debug_main_cpu debug_main_gpu


.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

debug_main_cpu: $(WICWIU_OBJS) main.o
	$(NVCC) $(CFLAGS) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

debug_main_gpu: $(WICWIU_OBJS) main.o
	$(NVCC) $(CFLAGS) $(DFLAGS) $(CUDA_ENABLE) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

main_cpu: $(WICWIU_OBJS) main.o
	$(NVCC) $(CFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

main_gpu: $(WICWIU_OBJS) main.o
	$(NVCC) $(CFLAGS) $(CUDA_ENABLE) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

clean:
	rm -rf *.o $(WICWIU_OBJS) main_cpu main_gpu debug_main_cpu debug_main_gpu
