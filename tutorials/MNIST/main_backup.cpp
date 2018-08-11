#include "net/my_CNN.h"
#include "net/my_NN.h"
#include "net/TransposedConv_Test.h"
#include "net/my_Resnet.h"
#include "MNIST_Reader.h"
#include <time.h>

#define BATCH             25
#define EPOCH             1000
#define LOOP_FOR_TRAIN    (60000 / BATCH)
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    //FILE *fptr;
    //fptr = fopen("relu_result.txt", "w");

    clock_t startTime, endTime;
    double  nProcessExcuteTime;

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 10, "label");
    //Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 784, "label");

    // ======================= Select net ===================
    //NeuralNetwork<float> *net = new TransposedConv_Test(x, label);
    NeuralNetwork<float> *net = new my_CNN(x, label);
    // NeuralNetwork<float> *net = new my_NN(x, label, isSLP);
    // NeuralNetwork<float> *net = new my_NN(x, label, isMLP);
    // NeuralNetwork<float> *net = Resnet14<float>(x, label);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

#ifdef __CUDNN__
    x->SetDeviceGPU();
    label->SetDeviceGPU();
    net->SetDeviceGPU();
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
            dataset->CreateTrainDataPair(BATCH);

            Tensor<float> *x_t = dataset->GetTrainFeedImage();
            //Tensor<float> *x_copy = new Tensor<float>(x_t);
            Tensor<float> *l_t = dataset->GetTrainFeedLabel();

#ifdef __CUDNN__
            x_t->SetDeviceGPU();
            l_t->SetDeviceGPU();
            //x_copy->SetDeviceGPU();

#endif  // __CUDNN__

            net->FeedInputTensor(2, x_t, l_t);
            //net->FeedInputTensor(2, x_t, x_copy);


            net->ResetParameterGradient();

            net->Training();

            train_accuracy    += net->GetAccuracy();

            train_avg_loss    += net->GetLoss();

            printf("\rTraining complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1)
                   /*nProcessExcuteTime*/);
            fflush(stdout);

            if (j % 100 == 99) std::cout << '\n';
        }
        endTime = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Testing ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInferencing();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            dataset->CreateTestDataPair(BATCH);

            Tensor<float> *x_t = dataset->GetTestFeedImage();
            Tensor<float> *l_t = dataset->GetTestFeedLabel();
            //Tensor<float> *x_copy = new Tensor<float>(x_t);

#ifdef __CUDNN__
            x_t->SetDeviceGPU();
            l_t->SetDeviceGPU();
            //x_copy->SetDeviceGPU();

#endif  // __CUDNN__

            net->FeedInputTensor(2, x_t, l_t);
            //net->FeedInputTensor(2, x_t, x_copy);
            net->Testing();

            test_accuracy += net->GetAccuracy();
            test_avg_loss += net->GetLoss();

            printf("\rTesting complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
            //fprintf(fptr,"\n\nepoch: %d, loss: %f, accuracy: %f", i, test_avg_loss / (j + 1), test_accuracy / (j + 1));
        }
        std::cout << "\n\n";
    }

    delete dataset;
    delete net;

    //fclose(fptr);
    return 0;
}
