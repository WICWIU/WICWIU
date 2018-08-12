#include "net/my_Resnet.h"
// #include "MNIST_Reader.h"
#include <time.h>

#define BATCH             25
#define EPOCH             1000
#define LOOP_FOR_TRAIN    (60000 / BATCH)
#define LOOP_FOR_TEST     (10000 / BATCH)
#define GPUID             0


int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 10, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = Resnet14<float>(x, label);

    // ======================= Prepare Data ===================
    // MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        // ======================= Training =======================
        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTraining();

        startTime = clock();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            // dataset->CreateTrainDataPair(BATCH);

            // Tensor<float> *x_t = dataset->GetTrainFeedImage();
            // Tensor<float> *l_t = dataset->GetTrainFeedLabel();

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);  // 추후 자동화 필요
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
            // std::cin >> temp;
            // net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            net->Training();
            // std::cin >> temp;
            train_accuracy += net->GetAccuracy();
            train_avg_loss += net->GetLoss();

            printf("\rTraining complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1)
                   /*nProcessExcuteTime*/);
            fflush(stdout);

            if (j % 100 == 99) std::cout << '\n';
        }
        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Testing ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInferencing();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            // dataset->CreateTestDataPair(BATCH);

            // Tensor<float> *x_t = dataset->GetTestFeedImage();
            // Tensor<float> *l_t = dataset->GetTestFeedLabel();

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            // net->FeedInputTensor(2, x_t, l_t);
            net->Testing();

            test_accuracy += net->GetAccuracy();
            test_avg_loss += net->GetLoss();

            printf("\rTesting complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << "\n\n";
    }

    // delete dataset;
    delete net;

    return 0;
}
