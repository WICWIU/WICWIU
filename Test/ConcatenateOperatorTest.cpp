#include "../WICWIU_src/NeuralNetwork.h"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 2, 2, 0.0, 0.1), "x");
    Tensorholder<float> *input1 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 3, 2, 2, 0.0, 0.1), "label");

    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    std::cout << input1->GetResult()->GetShape() << '\n';
    std::cout << input1->GetResult() << '\n';

    Operator<float> * concat = new ConcatenateChannelWise<float>(input0, input1);

    std::cout << concat->GetResult()->GetShape() << '\n';
    std::cout << concat->GetResult() << '\n';

    concat->ForwardPropagate();

    std::cout << concat->GetResult()->GetShape() << '\n';
    std::cout << concat->GetResult() << '\n';

    delete input0;
    delete input1;
    delete concat;

    return 0;
}
