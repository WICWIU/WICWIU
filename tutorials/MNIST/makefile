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
	../../WICWIU_src/Shape.cpp	\
	../../WICWIU_src/LongArray.cpp	\
	../../WICWIU_src/Tensor.cpp	\
	../../WICWIU_src/Operator.cpp	\
	../../WICWIU_src/LossFunction.cpp	\
	../../WICWIU_src/Optimizer.cpp	\
	../../WICWIU_src/Module.cpp	\
	../../WICWIU_src/NeuralNetwork.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

all: main


.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@

main: $(WICWIU_OBJS) main.o
	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

#main: main.o $(WICWIU_LIB)
#	$(LINKER) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_LIB) main.o


$(WICWIU_LIB): $(WICWIU_SRC)
	make -C ../..

clean:
	rm -rf *.o main

clean_all:
	rm -rf *.o main
	make clean -C ../..
