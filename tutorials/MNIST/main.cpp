#include "net/my_CNN.hpp"
#include "net/my_NN.hpp"
#include "net/my_FaceNetNN.hpp"
#include "net/my_Resnet.hpp"
#include "MNIST.hpp"
#include <time.h>

#define BATCH             120
#define EPOCH             100
#define LOOP_FOR_TRAIN    (60000 / BATCH)
#define LOOP_FOR_TEST     (10000 / BATCH)
#define GPUID             6

float* calDist(Operator<float> * pred, Operator<float> * ref);
int argMax(Tensor<float> * dist);
float knn(int k, Operator<float> * pred, Operator<float> * ref, Operator<float> * labelOfRef);
float GetAccuracy(int k, Operator<float> * pred, Operator<float> * labelOfPred, Operator<float> * ref, Operator<float> * labelOfRef);

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;
    // char filename[]      = "MNIST_parmas";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 10, "label");

    // ======================= Select net ===================
    // NeuralNetwork<float> *net = new my_CNN(x, label);
    // NeuralNetwork<float> *net = new my_NN(x, label, isSLP);
    // NeuralNetwork<float> *net = new my_NN(x, label, isMLP);
    // NeuralNetwork<float> *net = Resnet14<float>(x, label);
    NeuralNetwork<float> *net = new my_FaceNetNN(x, label);

#ifdef __CUDNN__
    // x->SetDeviceGPU(GPUID);
    // label->SetDeviceGPU(GPUID);
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    // ======================= Prepare Data ===================
    //MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();
    MNISTDataSet<float> *train_dataset = new MNISTDataSet<float>("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", TRAINING);
    DataLoader<float> * train_dataloader = new DataLoader<float>(train_dataset, BATCH, TRUE, 20, FALSE);

    MNISTDataSet<float> *test_dataset = new MNISTDataSet<float>("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", TESTING);
    DataLoader<float> * test_dataloader = new DataLoader<float>(test_dataset, BATCH, FALSE, 20, FALSE);

    // ======================= for KNN ===================
    std::cout << "KNN Reference" << '\n';
    Operator<float> *knn_ref = new ReShape<float>(net, 1, 1024, "KNN_REF");
    Operator<float> *ref_label = new ReShape<float>(label, 1, 10, "REF_label");

#ifdef __CUDNN__
    knn_ref->SetDeviceGPU(net->GetCudnnHandle(), GPUID);
    ref_label->SetDeviceGPU(net->GetCudnnHandle(), GPUID);

#endif  // __CUDNN__
    knn_ref->PrintInformation(0);
    ref_label->PrintInformation(0);

    float best_acc = 0;
    int   epoch    = 0;

    // @ When load parameters
    // net->Load(filename);

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
        // float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            //dataset->CreateTrainDataPair(BATCH);
            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            // printf("%d\r\n", temp->size());

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];
            delete temp;

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);  // 異뷀썑 ?먮룞???꾩슂
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
            // std::cin >> temp;
            net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            net->Train();
            // std::cin >> temp;

            // train_accuracy += net->GetAccuracy();
            train_avg_loss += net->GetLoss();
            // train_avg_loss += net->GetClassifierLoss();

            printf("\rTrain complete percentage is %d / %d -> loss : %f"/*, acc : %f"*/  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1)/*,
                   train_accuracy / (j + 1)*/
                   /*nProcessExcuteTime*/);
            fflush(stdout);
        }
        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();
        // create KNN reference
        std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
        // printf("%d\r\n", temp->size());

        Tensor<float> *x_t = (*temp)[0];
        Tensor<float> *l_t = (*temp)[1];
        delete temp;

#ifdef __CUDNN__
        x_t->SetDeviceGPU(GPUID);  // 異뷀썑 ?먮룞???꾩슂
        l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
        // std::cin >> temp;
        net->FeedInputTensor(2, x_t, l_t);
        net->Test();

#ifdef __CUDNN__
        knn_ref->ForwardPropagateOnGPU();
        ref_label->ForwardPropagateOnGPU();
#else
        knn_ref->ForwardPropagate();
        ref_label->ForwardPropagate();
#endif  // __CUDNN__

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            //dataset->CreateTestDataPair(BATCH);
            std::vector<Tensor<float> *> * temp =  test_dataloader->GetDataFromGlobalBuffer();
            // printf("%d\r\n", temp->size());

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];
            delete temp;

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            net->FeedInputTensor(2, x_t, l_t);
            net->Test();

            calDist(net, knn_ref);


            test_accuracy += GetAccuracy(1, net, label, knn_ref, ref_label);
            test_avg_loss += net->GetLoss();
            // test_avg_loss += net->GetClassifierLoss();
            printf("\rTest complete percentage is %d / %d -> loss : %f"/*, acc : %f"*/,
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1)/*,
                   test_accuracy / (j + 1)*/);
            fflush(stdout);
        }
        std::cout << "\n\n";

        // if ((best_acc < (test_accuracy / LOOP_FOR_TEST))) {
        //     net->Save(filename);
        // }
    }

    //delete dataset;
    delete net;
    delete train_dataloader;
    delete train_dataset;
    delete test_dataloader;
    delete test_dataset;

    return 0;
}

float* calDist(Operator<float> * pred, Operator<float> * ref){
    Tensor<float> *x = pred->GetResult();
    Tensor<float> *y = ref->GetResult();

    int timesize    = pred->GetResult()->GetTimeSize();
    int batchsize   = pred->GetResult()->GetBatchSize();
    int channelsize = pred->GetResult()->GetChannelSize();
    int rowsize     = pred->GetResult()->GetRowSize();
    int colsize     = pred->GetResult()->GetColSize();

    int ref_col     = ref->GetResult()->GetColSize();

    int ref_capacity = channelsize * rowsize * ref_col;
    int capacity    = channelsize * rowsize * colsize;

    float *res;

    for (int ba = 0, i = 0; ba < batchsize; ba++) {
        i = timesize * batchsize + ba;

        for (int j = 0, ref_index = 0; j < ref_capacity; j++) {
          for(int k = 0, index = 0; k < capacity; k++){
            index         = i * capacity + k;
            ref_index     = i * ref_capacity + j;
            std::cout << "1" << '\n';
            res[i] += (((*x)[index] - (*y)[ref_index]) * ((*x)[index] - (*y)[ref_index]));

            std::cout << "x: " << x << '\n';
            std::cout << "y: " << y << '\n';
            std::cout << "result: " << res <<'\n';
            int f;
            std::cin >> f;
          }
        }
      }

}

int maxArg(float* distList){
    return 0;
}

float knn(int k, Operator<float> * pred, Operator<float> * ref, Operator<float> * labelOfRef){
    return 0.f;
}

float GetAccuracy(int k, Operator<float> * pred, Operator<float> * labelOfPred, Operator<float> * ref, Operator<float> * labelOfRef){

    return 0.f;
}
