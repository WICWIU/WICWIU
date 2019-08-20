#include "net/my_CNN.hpp"
#include "net/my_NN.hpp"
#include "net/my_FaceNetNN.hpp"
#include "net/my_Resnet.hpp"
#include "MNIST.hpp"
#include "MNISTForSample.hpp"
#include "Sampler.hpp"
#include "SamplerForNeighbor.hpp"
#include "knn.hpp"
#include <time.h>
#include <map>
#include <iostream>
#include <fstream>

#define BATCH             120
#define EPOCH             120
#define LOOP_FOR_TRAIN    (60000 / BATCH)
#define LOOP_FOR_TEST     (10000 / BATCH)
#define GPUID             1
#define KNN_K             5

int main(int argc, char const *argv[]) {
    // int view[10][10]={0};
    // int total[10][10]={0};
    // std::ofstream file2;
    // std::ofstream file1;
    // std::ofstream file3;
    // std:: ofstream file4;
    // file4.open("accuracy1.txt", std::ios::out | std::ios::app);
    // file1.open("check1.txt", std::ios::out | std::ios::app);
    // file2.open("mat.txt", std::ios::out | std::ios::app);
    // file3.open("check1.txt", std::ios::out | std::ios::app);

    clock_t startTime, endTime;
    double  nProcessExcuteTime;
    int num100;
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
    net->PrintGraphInformation();

#ifdef __CUDNN__
    // x->SetDeviceGPU(GPUID);
    // label->SetDeviceGPU(GPUID);
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    // ======================= Prepare Data ===================
    //MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();
    MNISTDataSetForSample<float> *train_dataset = new MNISTDataSetForSample<float>("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", TRAINING);
    DataLoader<float> * train_dataloader = new Sampler<float>(10, train_dataset, BATCH, TRUE, 1, FALSE);

    MNISTDataSetForSample<float> *neighbors_dataset = new MNISTDataSetForSample<float>("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", TRAINING);
    DataLoader<float> * neighbors_dataloader = new SamplerForNeighbor<float>(10, neighbors_dataset, BATCH, FALSE, 1, FALSE);

    MNISTDataSet<float> *test_dataset = new MNISTDataSet<float>("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", TESTING);
    DataLoader<float> * test_dataloader = new DataLoader<float>(test_dataset, BATCH, FALSE, 1, FALSE);

    // ======================= for KNN ===================
    std::cout << "KNN Reference" << '\n';
    Operator<float> *knn_ref = new ReShape<float>(net, 1, 64, "KNN_REF");
    Operator<float> *ref_label = new ReShape<float>(label, 1, 10, "REF_label");

#ifdef __CUDNN__
    knn_ref->SetDeviceGPU(net->GetCudnnHandle(), GPUID);
    ref_label->SetDeviceGPU(net->GetCudnnHandle(), GPUID);

#endif  // __CUDNN__
    // knn_ref->PrintInformation(0);
    // ref_label->PrintInformation(0);

    float best_acc = 0;
    int   epoch    = 0;

    // @ When load parameters
    // net->Load(filename);

    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    for (int i = epoch + 1; i <= EPOCH; i++) {
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

        startTime = clock();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            //dataset->CreateTrainDataPair(BATCH);
            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            // printf("%d\r\n", temp->size());

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];

            // std::cout << '\n';
            // for(int n = 0; n < BATCH; n++){
            //     std::cout << onehot2label(n, l_t) << ' ';
            //     if(n == BATCH / 3 - 1 || n == BATCH / 3 * 2 - 1) std::cout << '\n';
            // }
            // std::cout << '\n';
            // std::cin >> num100;
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

            // train_accuracy += GetAccuracy(KNN_K, net, label, knn_ref, ref_label/*, view*/);
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
        // std::cin >> startTime;
        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();
        // create KNN reference
        std::vector<Tensor<float> *> *temp = neighbors_dataloader->GetDataFromGlobalBuffer();
        // printf("%d\r\n", temp->size());

        Tensor<float> *x_t = (*temp)[0];
        Tensor<float> *l_t = (*temp)[1];
        // std::cout << "neighbors "<< '\n';
        // for(int n = 0; n < BATCH; n++){
        //     std::cout << onehot2label(n, l_t) << ' ';
        //     if(n == BATCH / 3 - 1 || n == BATCH / 3 * 2 - 1) std::cout << '\n';
        // }
        // std::cout << '\n';
        delete temp;

#ifdef __CUDNN__
        x_t->SetDeviceGPU(GPUID);  // 異뷀썑 ?먮룞???꾩슂
        l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
        // std::cin >> temp;
        net->FeedInputTensor(2, x_t, l_t);
        net->Test();

        // std::cout << net -> GetResult() << '\n';
        // std::cout << label -> GetResult() << '\n';
        // cin >> num100;

        knn_ref -> ResetResult();
        ref_label -> ResetResult();

#ifdef __CUDNN__
        knn_ref->ForwardPropagateOnGPU();
        ref_label->ForwardPropagateOnGPU();
#else
        knn_ref->ForwardPropagate();
        ref_label->ForwardPropagate();
#endif  // __CUDNN__

        // std::cout << knn_ref->GetResult() << '\n';
        // std::cout << ref_label->GetResult() << '\n';
        // std::cin >> num100;

        file1<<"EPOCH"<<i<<endl;
        // file2<<"EPOCH"<<i<<endl;
        // file3<<"EPOCH"<<i<<endl;
        // file4<<"EPOCH"<<i<<endl;
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

            test_accuracy += GetAccuracy(KNN_K, net, label, knn_ref, ref_label/*, view*/);
            test_avg_loss += net->GetLoss();
            // test_avg_loss += net->GetClassifierLoss();
            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);

            // file4 << "Test complete percentage is " << j+1 << "/" << LOOP_FOR_TEST << " loss : " << test_avg_loss / (j+1) << "acc : " << test_accuracy / (j + 1) << '\n';
            // for(int i=0; i<10; i++)
            // {
            //   for(int j=0; j<10; j++)
            //     total[i][j] += view[i][j];
            // }
            // std::cin >> startTime;
        }

        // file4<<endl;
        std::cout << "\n\n";
      }
        // for(int i=0; i<10; i++)
        // {
        //   for(int j=0; j<10; j++)
        //     file2<< total[i][j] << ' ';
        //
        //   file2<<endl;
        // }
        // file2<<endl;
        // file2<<endl;
        // file1<<endl;
        //
        // for(int i=0; i<10; i++)
        // {
        //   for(int j=0; j<10; j++)
        //   {
        //     view[i][j]=0;
        //     total[i][j]=0;
        //   }
        // }
        // if ((best_acc < (test_accuracy / LOOP_FOR_TEST))) {
        //     net->Save(filename);
        // }


    //delete dataset;
    delete net;
    delete train_dataloader;
    delete train_dataset;
    delete test_dataloader;
    delete test_dataset;

    return 0;
}
