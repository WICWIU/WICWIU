#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    int inputSeq[] = { 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0};
	int desiredSeq[] = { 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };

    Tensor<float> *x_tensor = new Tensor<float>(6, 1, 1, 1, 6);
    Tensorholder<float> *x_holder = new Tensorholder<float>(x_tensor, "x");
    //Tensorholder<float> *x_holder = new Tensorholder<float>(6, 1, 1, 1, 6, "x(input)");

    Tensor<float> *label_tensor = new Tensor<float>(6, 1, 1, 1, 6);
    Tensorholder<float> *label_holder = new Tensorholder<float>(label_tensor, "label");
    //Tensorholder<float> *label_holder = new Tensorholder<float>(6, 1, 1, 1, 6, "label");

    for(int t = 0; t < 6; t++){
        for (int ba = 0; ba < 1; ba++) {
            for(int col = 0; col < 6; col++){
                (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
            }
        }
    }

    std::cout << x_holder->GetResult() << '\n';
    std::cout << label_holder->GetResult() << '\n';

    delete x_tensor;
    delete x_holder;
    delete label_tensor;
    delete label_holder;
    return 0;
}
