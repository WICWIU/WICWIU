.SUFFIXES = .cpp .o

WICWIU_LIB = lib/libwicwiu.a
CFLAGS = -O2 -std=c++11
ENABLE_CUDNN = -D__CUDNN__
#DFLAGS = -g -D__DEBUG__

LFLAGS = -lcudart -lcudnn -llibjpeg -lpthread

INCLUDE_PATH = -I../../JpegLib -I/usr/local/cuda/include -I/opt/libjpeg-turbo/include
LIB_PATH = -L. -L/usr/local/cuda/lib64

CC = g++
NVCC = nvcc
AR = ar

WICWIU_SRCS = \
	WICWIU_src/Shape.cpp	\
	WICWIU_src/LongArray.cpp	\
	WICWIU_src/Tensor.cpp	\
	WICWIU_src/Operator.cpp	\
	WICWIU_src/LossFunction.cpp	\
	WICWIU_src/Optimizer.cpp	\
	WICWIU_src/Layer.cpp	\
	WICWIU_src/ImageLoader.cpp	\
	WICWIU_src/NeuralNetwork.cpp

WICWIU_OBJS = ${WICWIU_SRCS:.cpp=.o}

all:	$(WICWIU_LIB) 

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(ENABLE_CUDNN) $(INCLUDE_PATH) $(LIB_PATH) -c $< -o $@


$(WICWIU_LIB): $(WICWIU_OBJS)
	$(AR) rcs $@ $(WICWIU_OBJS)

#main: $(WICWIU_OBJS) main.o
#	$(NVCC) $(CFLAGS) $(ENABLE_CUDNN) $(DFLAGS) $(LFLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $(WICWIU_OBJS) main.o

clean:
	rm -rf *.o $(WICWIU_OBJS) $(WICWIU_LIB)
