#include "net/my_SeqToSeq.hpp"
#include "net/my_AttentionSeqToSeq.hpp"
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

#define EMBEDDIM               256
#define HIDDENDIM              512
#define BATCH                  128 //256
#define EPOCH                  100
#define MAX_TRAIN_ITERATION    300
#define MAX_TEST_ITERATION     10
#define GPUID                  0

int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    // Train Dataset
    RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/kor-eng_train_90000.txt", "eng", "kor");
    // RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/kor_test.txt", "eng", "kor");
    translation_data->BuildVocab(TRAIN);
    DataLoader<float> * train_dataloader = new DataLoader<float>(translation_data, BATCH, FALSE, 1, FALSE);

    // Test Dataset
    TextDataset<float> *test_dataset = new RNNParalleledCorpusDataset<float>("Data/kor-eng_test_10000.txt", "kor", "eng");
    // TextDataset<float> *test_dataset = new RNNParalleledCorpusDataset<float>("Data/kor_test.txt", "kor", "eng");
    test_dataset->SetVocabs(translation_data);
    test_dataset->BuildVocab(INFERENCE);
    DataLoader<float> * test_dataloader = new DataLoader<float>(test_dataset, BATCH, FALSE, 1, TRUE);
    test_dataloader->StartProcess();


    int EncoderTime = translation_data->GetSrcVocabulary()->GetMaxSentenceLength();
    int DecoderTime = translation_data->GetTgtVocabulary()->GetMaxSentenceLength();
    int Srcvocab_size  = translation_data->GetSrcVocabulary()->GetNumberofUniqueWords();
    int Tgtvocab_size  = translation_data->GetTgtVocabulary()->GetNumberofUniqueWords();

 
    std::cout<<"Encoder Time : "<<EncoderTime<<'\n'<<"Decoder Time : "<<DecoderTime<<'\n'<<"Srcvocab_size : "<<Srcvocab_size<<'\n'<<"Tgtvocab_size : "<<Tgtvocab_size<<'\n';

    Tensorholder<float> *encoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(EncoderTime, BATCH, 1, 1, 1), "Encoder_input");
    Tensorholder<float> *decoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, 1), "Decoder_input");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, Tgtvocab_size), "label");
    Tensorholder<float> *encoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "EncoderLengths");
    Tensorholder<float> *decoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "DecoderLengths");

    NeuralNetwork<float> *net = new my_SeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, Tgtvocab_size, EMBEDDIM, HIDDENDIM, encoder_lengths_holder, decoder_lengths_holder);
    // NeuralNetwork<float> *net = new my_AttentionSeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, encoder_lengths_holder, decoder_lengths_holder, Tgtvocab_size, EMBEDDIM, HIDDENDIM);  


#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    map<int, string> *Srcindex2vocab = translation_data->GetSrcVocabulary()->GetIndex2Word();
    map<int, string> *Tgtindex2vocab = translation_data->GetTgtVocabulary()->GetIndex2Word();


    float best_acc = 0;
    int   epoch    = 0;

    train_dataloader->StartProcess();

    for (int i = epoch + 1; i < EPOCH; i++) {

        std::cout << "EPOCH : " << i << '\n';

        if ((i + 1) % 50 == 0) {
            std::cout << "Change learning rate!" << '\n';
            float lr = net->GetOptimizer()->GetLearningRate();
            net->GetOptimizer()->SetLearningRate(lr * 0.9);
        }

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            clock_t itrTime = clock();

            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();

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



            train_accuracy += net->GetAccuracy(Tgtvocab_size, d_l);
            train_avg_loss += net->GetLoss(d_l);
            

            clock_t itrEnd = clock();

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f , time : %f"   ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1), ((double)(itrEnd - itrTime)) / CLOCKS_PER_SEC
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

        std::cout << " Start Test" <<'\n';

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

            std::vector<Tensor<float> *> * temp =  test_dataloader->GetDataFromGlobalBuffer();

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


            EncoderTime = e_t->GetTimeSize();  
            Shape* eShape = e_t->GetShape();

            for(int en = 0; en < EncoderTime; en ++){
                std::cout<<Srcindex2vocab->at((*e_t)[Index5D(eShape, en, 0, 0, 0, 0)]);
                // fout<<Srcindex2vocab->at((*e_t)[Index5D(eShape, en, 0, 0, 0, 0)])<<" ";
            }
            std::cout<<'\n';

            net->FeedInputTensor(4, e_t, d_t, l_t, e_l);

            #ifdef __CUDNN__
                std::string result = net->SentenceTranslateOnGPU(Tgtindex2vocab, "translate_result.txt");
            #else
                std::string result = net->SentenceTranslateOnCPU(Tgtindex2vocab, "translate_result.txt");             
            #endif

            std::cout<<result;

            std::cout << "\n";

        }


    }

    delete net;

    return 0;
}
