#include "net/my_Resnet.h"
#include "net/my_CNN.h"
#include "CIFAR10Reader.h"
#include <time.h>
#include <unistd.h>

#define BATCH             25
#define EPOCH             1000
#define LOOP_FOR_TRAIN    (50000 / BATCH)
#define LOOP_FOR_TEST     (10000 / BATCH)
#define GPUID             0
#define LOG_LENGTH        1

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;

    char filename[] = "20180821-23-01-resnet34.b";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 3072, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 10, "label");

    // ======================= Select net ===================
    // NeuralNetwork<float> *net = new my_CNN(x, label);
    NeuralNetwork<float> *net = Resnet18<float>(x, label);
    // NeuralNetwork<float> *net = Resnet34<float>(x, label);
    net->PrintGraphInformation();

    // ======================= Prepare Data ===================
    CIFAR10Reader<float> *train_data_reader = new CIFAR10Reader<float>(BATCH, 500, TRUE);
    train_data_reader->UseNormalization(TRUE);
    train_data_reader->UseRandomCrop(4, 32);
    train_data_reader->UseRandomHorizontalFlip();
    train_data_reader->StartProduce();

    CIFAR10Reader<float> *test_data_reader = new CIFAR10Reader<float>(BATCH, 500, FALSE);
    test_data_reader->UseNormalization(TRUE, train_data_reader);
    test_data_reader->StartProduce();

    Tensor<float> **data = NULL;

    #ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);  // CUDNN ERROR
    #endif  // __CUDNN__

    float best_acc = 0.f;
    int   epoch    = 0;

    // @ When load parameters
    FILE *fp = fopen(filename, "rb");
    net->Load(fp);
    fread(&best_acc, sizeof(float), 1, fp);
    fread(&epoch,    sizeof(int),   1, fp);
    fclose(fp);

    printf("%s\n", filename);

    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        if ((i + 1) % 40 == 0) {
            std::cout << "Change learning rate!" << '\n';
            float lr = net->GetOptimizer()->GetLearningRate();
            net->GetOptimizer()->SetLearningRate(lr * 0.3);
            std::cout << "lr : " << lr * 0.5 << '\n';
        }
        // ======================= Training =======================
        float train_avg_accuracy = 0.f;
        float train_cur_accuracy = 0.f;
        float train_avg_loss     = 0.f;
        float train_cur_loss     = 0.f;

        net->SetModeTraining();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            startTime = clock();

            data = train_data_reader->GetDataFromBuffer();

    #ifdef __CUDNN__
            data[0]->SetDeviceGPU(GPUID);  // 추후 자동화 필요
            data[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__

            // std::cin >> temp;
            net->FeedInputTensor(2, data[0], data[1]);
            delete data;
            data = NULL;
            net->ResetParameterGradient();
            net->Training();

            train_cur_accuracy = net->GetAccuracy();
            train_cur_loss     = net->GetLoss();

            train_avg_accuracy += train_cur_accuracy;
            train_avg_loss     += train_cur_loss;

            endTime            = clock();
            nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;

            printf("\r%d / %d -> cur_loss : %0.4f, avg_loss : %0.4f, cur_acc : %0.5f, avg_acc : %0.5f, ct : %0.3f's / rt : %0.3f'm"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_cur_loss,
                   train_avg_loss / (j + 1),
                   train_cur_accuracy,
                   train_avg_accuracy / (j + 1),
                   nProcessExcuteTime,
                   nProcessExcuteTime * (LOOP_FOR_TRAIN - j - 1) / 60);
            fflush(stdout);

            // sleep(30);
            if (j % (LOOP_FOR_TRAIN / LOG_LENGTH) == (LOOP_FOR_TRAIN / LOG_LENGTH) - 1) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';
        // std::cout << net->GetResultOperator()->GetResult() << '\n';
        // int temp;
        // std::cin >> temp;

        // ======================= Accumulating =======================
    //     train_avg_accuracy = 0.f;
    //     train_cur_accuracy = 0.f;
    //     train_avg_loss     = 0.f;
    //     train_cur_loss     = 0.f;
    //
    //     net->SetModeAccumulating();
    //
    //     for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
    //         startTime = clock();
    //
    //         data = train_data_reader->GetDataFromBuffer();
    //
    // #ifdef __CUDNN__
    //         data[0]->SetDeviceGPU(GPUID);  // 추후 자동화 필요
    //         data[1]->SetDeviceGPU(GPUID);
    // #endif  // __CUDNN__
    //
    //         // std::cin >> temp;
    //         net->FeedInputTensor(2, data[0], data[1]);
    //         delete data;
    //         data = NULL;
    //         net->ResetParameterGradient();
    //         net->Testing();
    //         // std::cin >> temp;
    //         train_cur_accuracy = net->GetAccuracy();
    //         train_cur_loss     = net->GetLoss();
    //
    //         train_avg_accuracy += train_cur_accuracy;
    //         train_avg_loss     += train_cur_loss;
    //
    //         endTime            = clock();
    //         nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    //
    //         printf("\r%d / %d -> cur_loss : %0.4f, avg_loss : %0.4f, cur_acc : %0.5f, avg_acc : %0.5f, ct : %0.3f's / rt : %0.3f'm"  /*(ExcuteTime : %f)*/,
    //                j + 1, LOOP_FOR_TRAIN,
    //                train_cur_loss,
    //                train_avg_loss / (j + 1),
    //                train_cur_accuracy,
    //                train_avg_accuracy / (j + 1),
    //                nProcessExcuteTime,
    //                nProcessExcuteTime * (LOOP_FOR_TRAIN - j - 1) / 60);
    //         fflush(stdout);
    //
    //         // sleep(30);
    //         if (j % (LOOP_FOR_TRAIN / LOG_LENGTH) == (LOOP_FOR_TRAIN / LOG_LENGTH) - 1) {
    //             std::cout << '\n';
    //         }
    //     }
    //     std::cout << '\n';

        // ======================= Testing ======================
        float test_avg_accuracy = 0.f;
        float test_avg_loss     = 0.f;

        net->SetModeInferencing();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            data = test_data_reader->GetDataFromBuffer();

    #ifdef __CUDNN__
            data[0]->SetDeviceGPU(GPUID);  // 추후 자동화 필요
            data[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__

            net->FeedInputTensor(2, data[0], data[1]);
            delete data;
            data = NULL;
            net->Testing();



            test_avg_accuracy += net->GetAccuracy();
            test_avg_loss     += net->GetLoss();

            printf("\r%d / %d -> avg_loss : %0.4f, avg_acc : %0.4f, ct : %0.3f's / rt : %0.3f'm"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_avg_accuracy / (j + 1),
                   nProcessExcuteTime,
                   nProcessExcuteTime * (LOOP_FOR_TEST - j - 1) / 60);
            fflush(stdout);
        }

        if (best_acc < test_avg_accuracy / LOOP_FOR_TEST) {
            std::cout << "\nsave parameters...";
            FILE *fp = fopen(filename, "wb");
            net->Save(fp);
            best_acc = test_avg_accuracy / LOOP_FOR_TEST;
            fwrite(&best_acc, sizeof(float), 1, fp);
            fwrite(&i,        sizeof(int),   1, fp);
            fclose(fp);
            std::cout << "done" << "\n\n";
        } else std::cout << "\n\n";
    }


    train_data_reader->StopProduce();
    test_data_reader->StopProduce();

    delete train_data_reader;
    delete test_data_reader;
    delete net;

    return 0;
}
