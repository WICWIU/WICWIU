.SUFFIXES = .cpp .o

CFLAGS = -O2 -std=c++11

# if cuda device, cuda lib, or cuDNN is not installed, disable the following line
ENABLE_CUDNN = -D__CUDNN__

#DFLAGS = -g -D__DEBUG__


INCLUDE_PATH = -I/usr/local/cuda/include
LIB_PATH = -L. -L/usr/local/cuda/lib64


CC = g++
ifdef ENABLE_CUDNN
	LINKER = nvcc
	LFLAGS = -lcudart -lcudnn -lpthread
else
	LINKER = g++
	LFLAGS = -lpthread
endif


WICWIU_LIB = ../../lib/libwicwiu.a

WICWIU_SRCS = \
   ../../WICWIU_src/Shape.cpp   \
   ../../WICWIU_src/LongArray.cpp   \
   ../../WICWIU_src/Tensor.cpp   \
   ../../WICWIU_src/Operator.cpp   \
   ../../WICWIU_src/LossFunction.cpp   \
   ../../WICWIU_src/Optimizer.cpp   \
   ../../WICWIU_src/Module.cpp   \
   ../../WICWIU_src/NeuralNetwork.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

all: main_backup


.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

main_backup: $(WICWIU_OBJS) main_backup.o
	$(NVCC) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main_backup.o

clean:
	rm -rf *.o $(WICWIU_OBJS) main_backup
