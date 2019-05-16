#include "net/my_Resnet.hpp"
#include "net/my_Densenet.hpp"
// #include "ImageNetReader.hpp"
#include "ImageNet.hpp"
#include <time.h>
#include <ctime>
#include <unistd.h>
#include <string>

#define NUMBER_OF_CLASS               1000
#define BATCH                         30
#define EPOCH                         1000
#define GPUID                         7
#define LOG_LENGTH                    5
#define LEARNING_RATE_DECAY_RATE      0.1
#define LEARNING_RATE_DECAY_TIMING    10

int main(int argc, char const *argv[]) {
    time_t startTime;
    struct tm *curr_tm;
    double     nProcessExcuteTime;
    float mean[]   = { 0.485*255, 0.456*255, 0.406*255 };
    float stddev[] = { 0.229*255, 0.224*255, 0.225*255 };

    char filename[] = "ImageNet_parmas";

    //// create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 150528, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 1000, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = Resnet18<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = Resnet34<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = DenseNetLite<float>(x, label, NUMBER_OF_CLASS);
    net->PrintGraphInformation();

    // ======================= Prepare Data ===================
    vision::Compose *transform            = new vision::Compose({/*new vision::Normalize({ 0.485*255, 0.456*255, 0.406*255 }, { 0.229*255, 0.224*255, 0.225*255 }),*/ new vision::Resize(256), new vision::RandomCrop(224), new vision::HorizentalFlip(224) });
    ImageNetDataset<float> *train_dataset = new ImageNetDataset<float>("/mnt/ssd/Data/ImageNet", "ILSVRC2012_img_train256", 1000, transform);
    DataLoader<float> *train_dataloader   = new DataLoader<float>(train_dataset, BATCH, TRUE, 15, FALSE);

    ImageNetDataset<float> *test_dataset = new ImageNetDataset<float>("/mnt/ssd/Data/ImageNet", "ILSVRC2012_img_val256", 1000, transform);
    DataLoader<float> *test_dataloader   = new DataLoader<float>(test_dataset, BATCH, FALSE, 5, TRUE);

    // int len = ds->GetLength();
    // int len = ds->GetLength() / BATCH;
    // std::cout << "len: " << len << '\n';
    // std::vector<Tensor<float> *> *v;

    // startTime = time(NULL);
    // curr_tm   = localtime(&startTime);
    // std::cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec << "\'s\n" << std::endl;
    //
    // for (int i = 0; i < len; i++) {
    // std::cout << "\rbatch sample: " << i;
    //// v = ds->GetData(i);
    // v = dl->GetDataFromGlobalBuffer();
    // delete (*v)[0];
    // delete (*v)[1];
    // delete v;
    // }
    // std::cout << '\n';

    // startTime = time(NULL);
    // curr_tm   = localtime(&startTime);
    // std::cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec << "\'s" << std::endl;

    #ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);  // CUDNN ERROR
    #endif  // __CUDNN__

    float best_acc = 0.f;
    int   epoch    = 0;

    // @ When load parameters
    // std::cout << "Loading..." << '\n';
    // net->Load(filename);
    // std::cout << "Done!" << '\n';

    std::cout << "filename : " << filename << '\n';
    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    if (epoch / LEARNING_RATE_DECAY_TIMING) {
        float lr = net->GetOptimizer()->GetLearningRate();
        net->GetOptimizer()->SetLearningRate(lr * pow(0.1, (int)(epoch / LEARNING_RATE_DECAY_TIMING)));
        std::cout << "lr : " << lr * pow(LEARNING_RATE_DECAY_RATE, (int)(epoch / LEARNING_RATE_DECAY_TIMING)) << '\n';
    }

    int loop_for_train = train_dataset->GetLength() / BATCH;
    int loop_for_test  = test_dataset->GetLength() / BATCH;
    int log_len        = loop_for_train / LOG_LENGTH;

    epoch = 0;

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        startTime = time(NULL);
        curr_tm   = localtime(&startTime);
        std::cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec << "\'s" << std::endl;

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
        float train_avg_accuracy      = 0.f;
        float train_avg_top5_accuracy = 0.f;
        float train_cur_accuracy      = 0.f;
        float train_cur_top5_accuracy = 0.f;
        float train_avg_loss          = 0.f;
        float train_cur_loss          = 0.f;

        net->SetModeTrain();

        for (int j = 0; j < loop_for_train; j++) {
            std::vector<Tensor<float> *> *temp = train_dataloader->GetDataFromGlobalBuffer();

            train_dataset -> Tensor2Image("test/test_img_" + std::to_string(j) + ".jpeg", (*temp)[0], TRUE);

    #ifdef __CUDNN__
            (*temp)[0]->SetDeviceGPU(GPUID);
            (*temp)[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__
            net->FeedInputTensor(2, (*temp)[0], (*temp)[1]);
            delete temp;
            temp = NULL;
            net->ResetParameterGradient();
            net->Train();
            train_cur_accuracy = net->GetAccuracy(NUMBER_OF_CLASS);
            // printf("train_cur : %.f\n", train_cur_accuracy);
            // train_cur_top5_accuracy = net->GetTop5Accuracy(NUMBER_OF_CLASS);
            train_cur_loss = net->GetLoss();

            train_avg_accuracy      += train_cur_accuracy;
            train_avg_top5_accuracy += train_cur_top5_accuracy;
            train_avg_loss          += train_cur_loss;

            printf("\r%d / %d -> cur_loss : %0.4f, avg_loss : %0.5f, cur_acc : %0.5f, avg_acc : %0.5f"  /*(ExcuteTime : %f)*/,
                   j + 1, loop_for_train,
                   train_cur_loss,
                   train_avg_loss / (j + 1),
                   train_cur_accuracy,
                   train_avg_accuracy / (j + 1));
            fflush(stdout);

            if (j % log_len == log_len - 1) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';


        // ======================= Test ======================
        float test_avg_accuracy      = 0.f;
        float test_avg_top5_accuracy = 0.f;
        float test_avg_loss          = 0.f;

        net->SetModeInference();

        for (int j = 0; j < loop_for_test; j++) {
            std::vector<Tensor<float> *> *temp = test_dataloader->GetDataFromGlobalBuffer();

    #ifdef __CUDNN__
            (*temp)[0]->SetDeviceGPU(GPUID);
            (*temp)[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__
            net->FeedInputTensor(2, (*temp)[0], (*temp)[1]);
            delete temp;
            net->Test();

            test_avg_accuracy += net->GetAccuracy(NUMBER_OF_CLASS);
            // test_avg_top5_accuracy += net->GetTop5Accuracy(NUMBER_OF_CLASS);
            test_avg_loss += net->GetLoss();

            printf("\r%d / %d -> avg_loss : %0.4f, avg_acc : %0.4f"  /*(ExcuteTime : %f)*/,
                   j + 1, loop_for_test,
                   test_avg_loss / (j + 1),
                   test_avg_accuracy / (j + 1));
            fflush(stdout);
        }

        if (best_acc < test_avg_accuracy / loop_for_test) {
            std::cout << "\nsave parameters...";
            net->Save(filename);
            std::cout << "done" << "\n\n";
        } else std::cout << "\n\n";
    }

    delete test_dataloader;
    delete test_dataset;
    delete train_dataloader;
    delete train_dataset;
    delete transform;
    delete net;

    return 0;
}
