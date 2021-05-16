
#include "net/my_SeqToSeq.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <map>
#include "TextDataset.hpp"

using namespace std;

#define EMBEDDIM               64
#define HIDDENDIM              128
#define BATCH                  16
#define EPOCH                  10
#define MAX_TRAIN_ITERATION    1500
#define MAX_TEST_ITERATION     5
#define GPUID                  1


int main(int argc, char const *argv[]) {


    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/test.txt", "eng", "fra");
    // RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/eng-fra_short.txt", "eng", "fra");
    translation_data->BuildVocab();

    DataLoader<float> * train_dataloader = new DataLoader<float>(translation_data, BATCH, TRUE, 20, FALSE);

    int EncoderTime = translation_data->GetEncoderMaxTime();
    int DecoderTime = translation_data->GetDecoderMaxTime();
    int vocab_size  = translation_data->GetNumberofVocabs();

    std::cout<<"------------------------------------------------"<<'\n';
    std::cout<<"Encoder Time : "<<EncoderTime<<'\n'<<"Decoder Time : "<<DecoderTime<<'\n'<<"vocab_size : "<<vocab_size<<'\n';


    Tensorholder<float> *encoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(EncoderTime, BATCH, 1, 1, 1), "Encoder_input");
    Tensorholder<float> *decoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, 1), "Decoder_input");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, vocab_size), "label");
    Tensorholder<float> *encoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "EncoderLengths");
    Tensorholder<float> *decoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "DecoderLengths");

    NeuralNetwork<float> *net = new my_SeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, vocab_size, EMBEDDIM, HIDDENDIM, encoder_lengths_holder, decoder_lengths_holder);


#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

     map<int, string> *index2vocab = translation_data->GetpIndex2Vocab();


    float best_acc = 0;
    int   epoch    = 0;

    train_dataloader->StartProcess();
    // net->FeedInputTensor(2, x, label);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            // printf("%d\r\n", temp->size());

            Tensor<float> *e_t = (*temp)[0];
            Tensor<float> *d_t = (*temp)[1];
            Tensor<float> *l_t = (*temp)[2];
            Tensor<float> *e_l = (*temp)[3];
            Tensor<float> *d_l = (*temp)[4];
            delete temp;

#ifdef __CUDNN__
            e_t->SetDeviceGPU(GPUID);
            d_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
            e_l->SetDeviceGPU(GPUID);
            d_l->SetDeviceGPU(GPUID);
#endif  // __CUDNN__



            net->FeedInputTensor(5, e_t, d_t, l_t, e_l, d_l);
            net->ResetParameterGradient();

        #ifdef __CUDNN__
            net->seq2seqBPTTOnGPU(EncoderTime, DecoderTime);
        #else
            net->seq2seqBPTT(EncoderTime, DecoderTime);
        #endif


            train_accuracy += net->GetAccuracy(vocab_size, d_l);
            train_avg_loss += net->GetLoss(d_l);



            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1)
                 );

             fflush(stdout);

        }


        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" <<'\n';

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {


            //data중에서 test!
            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            Tensor<float> *e_t = (*temp)[0];
            Tensor<float> *d_t = (*temp)[1];
            Tensor<float> *l_t = (*temp)[2];
            Tensor<float> *e_l = (*temp)[3];
            // Tensor<float> *d_l = (*temp)[4];
            delete temp;

#ifdef __CUDNN__
            e_t->SetDeviceGPU(GPUID);
            e_l->SetDeviceGPU(GPUID);
            d_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__


            Shape* eShape = e_t->GetShape();
            for(int en = 0; en < EncoderTime; en ++){
                std::cout<<index2vocab->at((*e_t)[Index5D(eShape, en, 0, 0, 0, 0)])<<" ";
            }
            std::cout<<'\n';

            // //e_t = translation_data->GetTestData("We're not desperate yet.");
            net->FeedInputTensor(4, e_t, d_t, l_t, e_l);
            //

            map<int, string>* index2vocab = translation_data->GetpIndex2Vocab();

            #ifdef __CUDNN__
                net->SentenceTranslateOnGPU(index2vocab);
            #else
                net->SentenceTranslateOnCPU(index2vocab);
            #endif


              std::cout << "\n\n";
        }


    }

    delete net;

    return 0;
}
