#include "../WICWIU_src/NeuralNetwork.hpp"
#include "../WICWIU_src/Operator/Recurrent.hpp"

//class Recurrent;

int main(int argc, char const *argv[]) {
    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    char alphabet[] = "helo";
    int outDim = 4;

    char inputStr[] = "hell";
    char desiredStr[] = "ello";

    char seqLen = 4;
	int inputSeq[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	int desiredSeq[] = { 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    Tensor<float> *x_tensor = new Tensor<float>(4, 1, 1, 1, 4);
    Tensorholder<float> *x_holder = new Tensorholder<float>(x_tensor, "x");

    Tensor<float> *label_tensor = new Tensor<float>(4, 1, 1, 1, 4);
    Tensorholder<float> *label_holder = new Tensorholder<float>(label_tensor, "label");

    //RNN<float> *net = new my_RNN<float>(x_holder,label_holder);

    for(int t = 0; t < seqLen; t++){
        for (int ba = 0; ba < 1; ba++) {
            for(int col = 0; col < 4; col++){
                (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                printf("Index5D-x_tensor: %d\n", Index5D(x_tensor->GetShape(), t, ba, 0, 0, col));
                (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                printf("Index5D-label_tensor: %d\n", Index5D(x_tensor->GetShape(), t, ba, 0, 0, col));
            }
        }
    }

    printf("Getshape \n");
    std::cout << x_holder->GetResult()->GetShape() << '\n';
    printf("GetResult \n");
    std::cout << x_holder->GetResult() << '\n';

    printf("Getshape \n");
    std::cout << label_holder->GetResult()->GetShape() << '\n';
    printf("GetResult \n");
    std::cout << label_holder->GetResult() << '\n';

    endTime            = clock();
    nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

    //std::cout <<

    delete x_tensor;
    delete x_holder;
    delete label_tensor;
    delete label_holder;

    return 0;
}

