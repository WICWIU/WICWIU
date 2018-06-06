debug_main_cpu:
	g++ -g -o testing -std=c++11 -D__DEBUG__ main.cpp ../../WICWIU_src/Shape.cpp ../../WICWIU_src/Data.cpp ../../WICWIU_src/Tensor.cpp ../../WICWIU_src/Operator.cpp ../../WICWIU_src/LossFunction.cpp ../../WICWIU_src/Optimizer.cpp ../../WICWIU_src/Layer.cpp ../../WICWIU_src/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

debug_main_gpu:
	g++ -g -o testing -std=c++11 -D__CUDNN__ -D__DEBUG__ main.cpp ../../WICWIU_src/Shape.cpp ../../WICWIU_src/Data.cpp ../../WICWIU_src/Tensor.cpp ../../WICWIU_src/Operator.cpp ../../WICWIU_src/LossFunction.cpp ../../WICWIU_src/Optimizer.cpp ../../WICWIU_src/Layer.cpp ../../WICWIU_src/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_cpu:
	g++ -g -o testing -std=c++11 main.cpp ../../WICWIU_src/Shape.cpp ../../WICWIU_src/Data.cpp ../../WICWIU_src/Tensor.cpp ../../WICWIU_src/Operator.cpp ../../WICWIU_src/LossFunction.cpp ../../WICWIU_src/Optimizer.cpp ../../WICWIU_src/Layer.cpp ../../WICWIU_src/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_gpu:
	g++ -g -o testing -std=c++11 -D__CUDNN__ main.cpp ../../WICWIU_src/Shape.cpp ../../WICWIU_src/Data.cpp ../../WICWIU_src/Tensor.cpp ../../WICWIU_src/Operator.cpp ../../WICWIU_src/LossFunction.cpp ../../WICWIU_src/Optimizer.cpp ../../WICWIU_src/Layer.cpp ../../WICWIU_src/NeuralNetwork.cpp -lcudart -lcudnn -lpthread
