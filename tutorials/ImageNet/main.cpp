#include "net/my_Resnet.hpp"
#include "net/my_Densenet.hpp"
#include "ImageNetReader.hpp"
#include <time.h>
#include <ctime>
#include <unistd.h>

#define NUMBER_OF_CLASS               1000
#define BATCH                         30
#define EPOCH                         1000
#define LOOP_FOR_TRAIN                (1281144 / BATCH)
#define LOOP_FOR_ACCUM                (10000 / BATCH) * 10
#define LOOP_FOR_TEST                 (50000 / BATCH)
#define GPUID                         0
#define LOG_LENGTH                    1
#define LEARNING_RATE_DECAY_RATE      0.1
#define LEARNING_RATE_DECAY_TIMING    10

int main(int argc, char const *argv[]) {
    time_t startTime;
    struct tm *curr_tm;
    double     nProcessExcuteTime;
    float mean[]   = { 0.485, 0.456, 0.406 };
    float stddev[] = { 0.229, 0.224, 0.225 };

    char filename[]      = "ImageNet_parmas";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 150528, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 1000, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = Resnet18<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = Resnet34<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = DenseNetLite<float>(x, label, NUMBER_OF_CLASS);
    net->PrintGraphInformation();

    // ======================= Prepare Data ===================
    ImageNetDataReader<float> *train_data_reader = new ImageNetDataReader<float>(BATCH, 25, TRUE);
    train_data_reader->UseNormalization(TRUE, mean, stddev);
    train_data_reader->UseRandomHorizontalFlip();
    // train_data_reader->UseRandomVerticalFlip();

    ImageNetDataReader<float> *test_data_reader = new ImageNetDataReader<float>(BATCH, 25, FALSE);
    test_data_reader->UseNormalization(TRUE, mean, stddev);

    train_data_reader->StartProduce();
    test_data_reader->StartProduce();

    Tensor<float> **data = NULL;

#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);  // CUDNN ERROR
#endif  // __CUDNN__

    float best_acc = 0.f;
    int   epoch    = 0;

    // @ When load parameters
    std::cout << "Loading..." << '\n';
    net->Load();
    std::cout << "Done!" << '\n';

    std::cout << "filename : " << filename << '\n';
    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    if (epoch / LEARNING_RATE_DECAY_TIMING) {
        float lr = net->GetOptimizer()->GetLearningRate();
        net->GetOptimizer()->SetLearningRate(lr * pow(0.1, (int)(epoch / LEARNING_RATE_DECAY_TIMING)));
        std::cout << "lr : " << lr * pow(LEARNING_RATE_DECAY_RATE, (int)(epoch / LEARNING_RATE_DECAY_TIMING)) << '\n';
    }

    // net->GetOptimizer()->SetLearningRate(0.000001);
    epoch = 0;

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        startTime = time(NULL);
        curr_tm   = localtime(&startTime);
        cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec << "\'s" << endl;

        if (i % LEARNING_RATE_DECAY_TIMING == 0) {
            std::cout << "Change learning rate!" << '\n';
            float lr = net->GetOptimizer()->GetLearningRate();
            net->GetOptimizer()->SetLearningRate(lr * LEARNING_RATE_DECAY_RATE);
            std::cout << "lr : " << lr * LEARNING_RATE_DECAY_RATE << '\n';
        } else {
            float lr = net->GetOptimizer()->GetLearningRate();
            std::cout << "lr : " << lr << '\n';
        }
        // ======================= Train =======================
    //     float train_avg_accuracy      = 0.f;
    //     float train_avg_top5_accuracy = 0.f;
    //     float train_cur_accuracy      = 0.f;
    //     float train_cur_top5_accuracy = 0.f;
    //     float train_avg_loss          = 0.f;
    //     float train_cur_loss          = 0.f;
    //
    //     net->SetModeTrain();
    //
    //     for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
    //         data = train_data_reader->GetDataFromBuffer();
    //
    // #ifdef __CUDNN__
    //         data[0]->SetDeviceGPU(GPUID);  // 추후 자동화 필요
    //         // std::cout << data[0]->GetShape() << '\n';
    //         data[1]->SetDeviceGPU(GPUID);
    //         // std::cout << data[1]->GetShape() << '\n';
    // #endif  // __CUDNN__
    //
    //         // std::cin >> temp;
    //         // std::cout << "test" << '\n';
    //         net->FeedInputTensor(2, data[0], data[1]);
    //         // std::cout << "test" << '\n';
    //         delete data;
    //         data = NULL;
    //         net->ResetParameterGradient();
    //         // std::cout << "test" << '\n';
    //         net->Train();
    //         // std::cout << "test" << '\n';
    //         // std::cin >> temp;
    //         train_cur_accuracy      = net->GetAccuracy(NUMBER_OF_CLASS);
    //         train_cur_top5_accuracy = net->GetTop5Accuracy(NUMBER_OF_CLASS);
    //         train_cur_loss          = net->GetLoss();
    //
    //         train_avg_accuracy      += train_cur_accuracy;
    //         train_avg_top5_accuracy += train_cur_top5_accuracy;
    //         train_avg_loss          += train_cur_loss;
    //
    //         printf("\r%d / %d -> cur_loss : %0.4f, avg_loss : %0.4f, cur_acc : %0.5f, avg_acc : %0.5f, cur_top5_acc : %0.5f, avg_top5_acc : %0.5f"  /*(ExcuteTime : %f)*/,
    //                j + 1, LOOP_FOR_TRAIN,
    //                train_cur_loss,
    //                train_avg_loss / (j + 1),
    //                train_cur_accuracy,
    //                train_avg_accuracy / (j + 1),
    //                train_cur_top5_accuracy,
    //                train_avg_top5_accuracy / (j + 1));
    //         fflush(stdout);
    //
    //         // if (train_cur_accuracy > train_cur_top5_accuracy) {
    //         // std::cout << "anomaly" << '\n';
    //         // int temp  = 0;
    //         // std::cin >> temp;
    //         // }
    //         // sleep(30);
    //         if (j % (LOOP_FOR_TRAIN / LOG_LENGTH) == (LOOP_FOR_TRAIN / LOG_LENGTH) - 1) {
    //             std::cout << '\n';
    //         }
    //     }
    //     std::cout << '\n';


        // ======================= Test ======================
        float test_avg_accuracy      = 0.f;
        float test_avg_top5_accuracy = 0.f;
        float test_avg_loss          = 0.f;

        net->SetModeInference();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            data = test_data_reader->GetDataFromBuffer();

    #ifdef __CUDNN__
            data[0]->SetDeviceGPU(GPUID);  // 추후 자동화 필요
            data[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__

            net->FeedInputTensor(2, data[0], data[1]);
            delete data;
            data = NULL;
            net->Test();

            test_avg_accuracy      += net->GetAccuracy(NUMBER_OF_CLASS);
            test_avg_top5_accuracy += net->GetTop5Accuracy(NUMBER_OF_CLASS);
            test_avg_loss          += net->GetLoss();

            printf("\r%d / %d -> avg_loss : %0.4f, avg_acc : %0.4f, avg_top5_acc : %0.4f"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_avg_accuracy / (j + 1),
                   test_avg_top5_accuracy / (j + 1));
            fflush(stdout);
        }

        // if (best_acc < test_avg_accuracy / LOOP_FOR_TEST) {
        //     std::cout << "\nsave parameters...";
        //     net->Save();
        //
        //     // FILE *fp_info = fopen(filename_info, "wb");
        //     // best_acc = test_avg_accuracy / LOOP_FOR_TEST;
        //     // fwrite(&best_acc, sizeof(float), 1, fp_info);
        //     // fwrite(&i,        sizeof(int),   1, fp_info);
        //     // fclose(fp_info);
        //
        //     std::cout << "done" << "\n\n";
        // } else std::cout << "\n\n";
    }

    train_data_reader->StopProduce();
    test_data_reader->StopProduce();

    delete train_data_reader;
    delete test_data_reader;
    delete net;

    return 0;
}
