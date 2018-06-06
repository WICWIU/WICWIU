debug_main_cpu:
	g++ -g -o testing -std=c++11 -D__DEBUG__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/Layer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

debug_main_gpu:
	g++ -g -o testing -std=c++11 -D__CUDNN__ -D__DEBUG__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/Layer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_cpu:
	g++ -g -o testing -std=c++11 main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/Layer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_gpu:
	g++ -g -o testing -std=c++11 -D__CUDNN__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/Layer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread
