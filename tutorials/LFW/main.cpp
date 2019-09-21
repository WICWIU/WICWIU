#include "net/my_Resnet.hpp"
#include "net/my_Densenet.hpp"
#include "net/my_Facenet.hpp"
#include "LFW.hpp"
#include "LFWReader.hpp"
#include "Sampler.hpp"
#include "SamplerForNeighbor.hpp"
#include "knn.hpp"
#include <time.h>
#include <ctime>
#include <unistd.h>

#define NUMBER_OF_CLASS               1680
#define BATCH                         45
#define EPOCH                         1000
#define GPUID                         1
#define LOG_LENGTH                    1
#define LEARNING_RATE_DECAY_RATE      0.1
#define LEARNING_RATE_DECAY_TIMING    10
#define KNN_K                         1

int main(int argc, char const *argv[]) {
    time_t startTime;
    struct tm *curr_tm;
    double     nProcessExcuteTime;
    float mean[]   = { 0.485, 0.456, 0.406 };
    float stddev[] = { 0.229, 0.224, 0.225 };

    char filename[] = "LFW_parmas";
    int num = 0;

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 150528, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, NUMBER_OF_CLASS, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = new my_FaceNet<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = Resnet34<float>(x, label, NUMBER_OF_CLASS);
    // NeuralNetwork<float> *net = DenseNetLite<float>(x, label, NUMBER_OF_CLASS);
    net->PrintGraphInformation();

    #ifdef __CUDNN__
        net->SetDeviceGPU(GPUID);  // CUDNN ERROR
    #endif  // __CUDNN__

    // std::cout << "/* Prepare Data */" << '\n';
    // ======================= Prepare Data ===================
    vision::Compose *transform = new vision::Compose({new vision::Resize(224)});
    LFWDatasetForSample<float> *train_dataset = new LFWDatasetForSample<float>("./data", "lfw_split", NUMBER_OF_CLASS, transform);
    DataLoader<float> *train_dataloader = new Sampler<float>(NUMBER_OF_CLASS, train_dataset, BATCH, TRUE, 1, FALSE);

    LFWDatasetForSample<float> *neighbors_dataset = new LFWDatasetForSample<float>("./data", "lfw_split", NUMBER_OF_CLASS, transform);
    DataLoader<float> *neighbors_dataloader = new SamplerForNeighbor<float>(NUMBER_OF_CLASS, neighbors_dataset, NUMBER_OF_CLASS, FALSE, 1, FALSE);

    LFWDataset<float> *test_dataset = new LFWDataset<float>("./data", "lfw_split", NUMBER_OF_CLASS, transform);
    DataLoader<float> *test_dataloader = new DataLoader<float>(test_dataset, 30, FALSE, 1, FALSE);

    // ================== for KNN =============================
    Operator<float> *knn_ref = new ReShape<float>(net, 1, 128, "KNN_REF");
    Operator<float> *ref_label = new ReShape<float>(label, 1, NUMBER_OF_CLASS, "REF_label");

#ifdef __CUDNN__
    knn_ref->SetDeviceGPU(net->GetCudnnHandle(), GPUID);
    ref_label->SetDeviceGPU(net->GetCudnnHandle(), GPUID);

#endif

    float best_acc = 0.f;
    int   epoch    = 0;
    int   loop_for_train = train_dataset -> GetLength() / BATCH;
    int   loop_for_test = 30;
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

    // net->GetOptimizer()->SetLearningRate(0.000001);
    epoch = 0;

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        startTime = time(NULL);
        curr_tm   = localtime(&startTime);
        std::cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec << "\'s" << '\n';

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
            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            // train_dataset -> Tensor2Image("test_image.jpeg", (*temp)[0], TRUE);

    #ifdef __CUDNN__
            (*temp)[0]->SetDeviceGPU(GPUID);  // 異뷀썑 ?먮룞???꾩슂
            // std::cout << data[0]->GetShape() << '\n';
            (*temp)[1]->SetDeviceGPU(GPUID);
            // std::cout << data[1]->GetShape() << '\n';
    #endif  // __CUDNN__

            // std::cout << '\n';
            // for(int n = 0; n < BATCH; n++){
            //     std::cout << onehot2label(n, (*temp)[1]) << ' ';
            //     if(n == BATCH / 3 - 1 || n == BATCH / 3 * 2 - 1) std::cout << '\n';
            // }
            // std::cout << '\n';
            // std::cin >> num;

            net->FeedInputTensor(2, (*temp)[0], (*temp)[1]);

            delete temp;
            temp = NULL;
            net->ResetParameterGradient();

            net->Train();


            // train_cur_accuracy      = net->GetAccuracy(NUMBER_OF_CLASS);
            // train_cur_top5_accuracy = net->GetTop5Accuracy(NUMBER_OF_CLASS);
            train_cur_loss          = net->GetLoss();

            // train_avg_accuracy      += train_cur_accuracy;
            // train_avg_top5_accuracy += train_cur_top5_accuracy;
            // train_avg_loss          += train_cur_loss;

            printf("\r%d / %d -> cur_loss : %0.4f",
                   j + 1, loop_for_train,
                   train_cur_loss);
            fflush(stdout);

            if (j % (loop_for_train / LOG_LENGTH) == (loop_for_train / LOG_LENGTH) - 1) {
                std::cout << '\n';
            }
        }
        std::cout << '\n';

        // ======================= Test ======================
        float test_avg_accuracy      = 0.f;
        float test_avg_top5_accuracy = 0.f;
        float test_avg_loss          = 0.f;

        net->SetModeInference();

        std::vector<Tensor<float> *> *temp = neighbors_dataloader->GetDataFromGlobalBuffer();

  #ifdef __CUDNN__
      (*temp)[0] -> SetDeviceGPU(GPUID);
      (*temp)[1] -> SetDeviceGPU(GPUID);
  #endif
      
      net -> FeedInputTensor(2, (*temp)[0], (*temp)[1]);
      delete temp;
      temp = NULL;

      net -> Test();

      knn_ref -> ResetResult();
      ref_label -> ResetResult();

  #ifdef __CUDNN__
      knn_ref -> ForwardPropagateOnGPU();
      ref_label -> ForwardPropagateOnGPU();

  #else
      knn_ref -> ForwardPropagate();
      ref_label -> ForwardPropagate();
  #endif

        for (int j = 0; j < loop_for_test; j++) {
            std::vector<Tensor<float> *> * temp =  test_dataloader->GetDataFromGlobalBuffer();

    #ifdef __CUDNN__
            (*temp)[0]->SetDeviceGPU(GPUID);  // 異뷀썑 ?먮룞???꾩슂
            (*temp)[1]->SetDeviceGPU(GPUID);
    #endif  // __CUDNN__

            net->FeedInputTensor(2, (*temp)[0], (*temp)[1]);
            delete temp;
            temp = NULL;
            net->Test();

            test_avg_accuracy      += GetAccuracy(KNN_K, net, label, knn_ref, ref_label);
            // test_avg_top5_accuracy += net->GetTop5Accuracy(NUMBER_OF_CLASS);
            test_avg_loss          += net->GetLoss();

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

    delete net;
    delete train_dataloader;
    delete train_dataset;
    delete test_dataloader;
    delete test_dataset;

    return 0;
}
