#include "net/my_CNN.hpp"
#include "net/my_NN.hpp"
#include "net/my_Resnet.hpp"
#include "MNIST_Reader.hpp"
#include <time.h>

#define BATCH             100
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
    NeuralNetwork<float> *net = new my_CNN(x, label);
    // NeuralNetwork<float> *net = new my_NN(x, label, isSLP);
    // NeuralNetwork<float> *net = new my_NN(x, label, isMLP);
    // NeuralNetwork<float> *net = Resnet14<float>(x, label);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *train_dataset = CreateMNISTTrainDataSet<float>(10, 100, GPUID);
    train_dataset->StartProduce();
    MNISTDataSet<float> *test_dataset  = CreateMNISTTestDataSet<float>(10, 100, GPUID);
    test_dataset->StartProduce();
    Tensor<float> **data = NULL;

#ifdef __CUDNN__
    // x->SetDeviceGPU(GPUID);
    // label->SetDeviceGPU(GPUID);
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    float best_acc = 0;
    int   epoch    = 0;

    // @ When load parameters
    // FILE *fp = fopen("parameters.b", "rb");
    // net->Load(fp);
    // fread(&best_acc, sizeof(float), 1, fp);
    // fread(&epoch,    sizeof(int),   1, fp);
    // fclose(fp);

    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        if ((i + 1) % 50 == 0) {
            std::cout << "Change learning rate!" << '\n';
            float lr = net->GetOptimizer()->GetLearningRate();
            net->GetOptimizer()->SetLearningRate(lr * 0.1);
        }

        // ======================= Train =======================
        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        nProcessExcuteTime = 0;
        startTime = clock();
        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            data = train_dataset->GetDataFromBuffer();
            Tensor<float> *x_t = data[0];
            Tensor<float> *l_t = data[1];
            delete data;
            data = NULL;

// #ifdef __CUDNN__
//             x_t->SetDeviceGPU(GPUID);  // 추후 자동화 필요
//             l_t->SetDeviceGPU(GPUID);
// #endif  // __CUDNN__
            // std::cin >> temp;
            net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            net->Train();
            // std::cin >> temp;
            train_accuracy += net->GetAccuracy();
            train_avg_loss += net->GetLoss();

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1)
                   /*nProcessExcuteTime*/);
            fflush(stdout);
        }
        endTime = clock();
        nProcessExcuteTime += ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            data = test_dataset->GetDataFromBuffer();
            Tensor<float> *x_t = data[0];
            Tensor<float> *l_t = data[1];
            delete data;
            data = NULL;

// #ifdef __CUDNN__
//             x_t->SetDeviceGPU(GPUID);
//             l_t->SetDeviceGPU(GPUID);
// #endif  // __CUDNN__

            net->FeedInputTensor(2, x_t, l_t);
            net->Test();

            test_accuracy += net->GetAccuracy();
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << "\n\n";

        if ((best_acc < (test_accuracy / LOOP_FOR_TEST))) {
            std::cout << "save parameters...";
            FILE *fp = fopen("parameters.b", "wb");
            net->Save(fp);
            best_acc = (test_accuracy / LOOP_FOR_TEST);
            fwrite(&best_acc, sizeof(float), 1, fp);
            fwrite(&i,        sizeof(int),   1, fp);
            fclose(fp);
            std::cout << "done" << "\n\n";
        }
    }

    delete train_dataset;
    delete test_dataset;
    delete net;

    return 0;
}
