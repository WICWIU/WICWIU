#include "net/my_RNN.hpp"
#include "net/my_LSTM.hpp"
#include "net/my_GRU.hpp"

#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#include "TextDataset.hpp"

using namespace std;

#define EMBED_DIM               64
#define HIDDEN_DIM              128
#define TIME                   100 //400
#define BATCH                  2 //32            //seqLength * batchsize < number of word!
#define EPOCH                  10
#define MAX_TRAIN_ITERATION    500
#define MAX_TEST_ITERATION     10
#define GPUID                  0


int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    RNNWordLevelDataset<float>* rnnWordData = new RNNWordLevelDataset<float>("Data/test.txt", TIME);

    rnnWordData->BuildVocab();

    std::cout<<"vocab size : "<<rnnWordData->GetNumberofVocabs()<<'\n';
    std::cout<<"word number : "<<rnnWordData->GetNumberofWords()<<'\n';

    DataLoader<float> * trainDataloader = new DataLoader<float>(rnnWordData, BATCH, TRUE, 20, FALSE);

    int vocabSize = rnnWordData->GetNumberofVocabs();
    Tensorholder<float> *xHolder = new Tensorholder<float>(TIME, BATCH, 1, 1, 1, "x");
    Tensorholder<float> *labelHolder = new Tensorholder<float>(TIME, BATCH, 1, 1, vocabSize, "label");

    // NeuralNetwork<float> *net = new my_RNN(xHolder,labelHolder, vocabSize, EMBED_DIM, HIDDEN_DIM);
    // NeuralNetwork<float> *net = new my_LSTM(xHolder,labelHolder, vocabSize, EMBED_DIM, HIDDEN_DIM);
    NeuralNetwork<float> *net = new my_GRU(xHolder,labelHolder, vocabSize, EMBED_DIM, HIDDEN_DIM);


#ifdef __CUDNN__
    // x->SetDeviceGPU(GPUID);
    // label->SetDeviceGPU(GPUID);`
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

    trainDataloader->StartProcess();


    for (int i = epoch + 1; i < EPOCH; i++) {

        float trainAccuracy = 0.f;
        float trainAvgLoss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            std::vector<Tensor<float> *> * temp =  trainDataloader->GetDataFromGlobalBuffer();

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];
            delete temp;


#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            net->BPTT(TIME);

            trainAccuracy += net->GetAccuracy(vocabSize);
            trainAvgLoss += net->GetLoss();


            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   trainAvgLoss /  (j + 1),
                   trainAccuracy / (j + 1)
                 );

             fflush(stdout);

        }


        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ==============Sentence Generation Test==================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" <<'\n';

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {


// #ifdef __CUDNN__
//             x_t->SetDeviceGPU(GPUID);
//             l_t->SetDeviceGPU(GPUID);
// #endif  // __CUDNN__

            int startIndex = 0;

            map<int, string>* Index2Vocab = rnnWordData->GetpIndex2Vocab();
            map<string, int>* Vocab2Index = rnnWordData->GetpVocab2Index();

            std::cout<<'\n'<<"starting with word first"<<'\n';
            startIndex = Vocab2Index->at("first");
            net->GenerateSentence(TIME, Index2Vocab, startIndex, vocabSize);

            // //fflush(stdout);
        }

    }

    delete net;
    delete trainDataloader;
    delete rnnWordData;


    return 0;
}
