/*g++ -g -o testing -std=c++11 main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective_.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork_.cpp*/

#include "net/my_CNN.h"
#include "net/my_NN.h"
#include "MNIST_Reader.h"
#include <time.h>

#define BATCH             50
#define EPOCH             100
#define LOOP_FOR_TRAIN    (60000 / BATCH)
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 10, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = new my_CNN(x, label);
    // NeuralNetwork<float> *net = new my_NN(x, label, isSLP);
    // NeuralNetwork<float> *net = new my_NN(x, label, isMLP);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

    net->PrintGraphShape();

#if __CUDNN__
    x->SetDeviceGPU();
    label->SetDeviceGPU();
    net->SetDeviceGPU();
#endif  // __CUDNN__

    // pytorch check하기
    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Training =======================
        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTraining();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);
            x->SetTensor(dataset->GetTrainFeedImage());
            label->SetTensor(dataset->GetTrainFeedLabel());

            startTime = clock();

            net->ResetParameterGradient();
            net->Training();

            endTime = clock();

            train_accuracy    += net->GetAccuracy();
            train_avg_loss    += net->GetLoss();
            nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;

            printf("\rTraining complete percentage is %d / %d -> loss : %f, acc : %f (ExcuteTime : %f)",
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1),
                   nProcessExcuteTime);
            fflush(stdout);

            if (j % 100 == 99) std::cout << '\n';
        }
        std::cout << '\n';

        // float accum_accuracy = 0.f;
        // float accum_avg_loss = 0.f;
        //
        // net->SetModeAccumulating();
        //
        // for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
        // dataset->CreateTrainDataPair(BATCH);
        // x->SetTensor(dataset->GetTrainFeedImage());
        // label->SetTensor(dataset->GetTrainFeedLabel());
        //
        // net->Testing();
        // accum_accuracy += net->GetAccuracy();
        // accum_avg_loss += net->GetLoss();
        //
        // printf("\rAccumulating complete percentage is %d / %d -> loss : %f, acc : %f",
        // j + 1, LOOP_FOR_TRAIN,
        // accum_avg_loss / (j + 1),
        // accum_accuracy / (j + 1));
        // fflush(stdout);
        // }
        // std::cout << '\n';

        // Caution!
        // Actually, we need to split training set between two set for training set and validation set
        // but in this example we do not above action.
        // ======================= Testing ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInferencing();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            dataset->CreateTestDataPair(BATCH);
            x->SetTensor(dataset->GetTestFeedImage());
            label->SetTensor(dataset->GetTestFeedLabel());

            net->Testing();
            test_accuracy += net->GetAccuracy();
            test_avg_loss += net->GetLoss();

            printf("\rTesting complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';
    }

    // we need to save best weight and bias when occur best acc on test time
    delete dataset;
    delete net;

    return 0;
}
