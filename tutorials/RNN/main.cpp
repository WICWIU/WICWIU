#include "net/my_RNN.hpp"
//#include "net/my_SeqToSeq.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더
#include "FinalTextDataset.hpp"
//#include "WordTextDataset.hpp"
//#include "BatchTextDataSet.hpp"
//#include "WordTextDataSet.hpp"
//#include "TextDataset2.hpp"

using namespace std;

#define EMBEDDIM               64
#define TIME                   100
#define BATCH                  4         //seqLength * batchsize < number of word!
#define EPOCH                  4
#define MAX_TRAIN_ITERATION    200   // (60000 / BATCH)
#define MAX_TEST_ITERATION     1   // (10000 / BATCH)
#define GPUID                  2



int main(int argc, char const *argv[]) {


    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    //WordTextDataset<float> *dataset = new WordTextDataset<float>("Data/Longtest.txt", ONEHOT);    //word 단위!
    //TextDataset<float> *dataset = new TextDataset<float>("Data/test20_2.txt", 100, ONEHOT);                         //TIME : 20, BATCH = 1
    //TextDataset<float> *dataset = new TextDataset<float>("Data/test566.txt", 100, ONEHOT);                          //TIME : 100  , batch = 5
    //TextDataset<float> *dataset = new TextDataset<float>("Data/last2.txt", 100, ONEHOT);     //char단위         //TIME : 500    , batch = 10      dataset 바꿀려고하면 TIME 도 바꿔줘야됨!!! batch도 맞춰줘야 되냐.....

    //DataLoader<float> * train_dataloader = new DataLoader<float>(dataset, BATCH, TRUE, 20, FALSE);

    // //char 단위
    // int Text_length = dataset->GetTextLength();
    // int vocab_size = dataset->GetVocabSize();
    //
    // std::cout<<"파일 길이 : "<<Text_length<<" vocab 길이 : "<<vocab_size<<'\n';
    // //char 단위
    //
    // //word 단위로 해보자!!!!!!!!!!!
    // //int Text_length      = dataset->GetWordNum();
    // //int vocab_size       = dataset->GetVocabSize();
    //
    // std::cout<<"Train 파일에 있는 단어 개수 : "<<Text_length<<" / vocab 개수 : "<<vocab_size<<'\n';
    // //word 단위로 해보자!!!!!!!



    //FinalTextDataset 2021/3/29
    // RNNWordLevelDataset<float>* rnnWordData = new RNNWordLevelDataset<float>("Data/last.txt", TIME);
    RNNWordLevelDataset<float>* rnnWordData = new RNNWordLevelDataset<float>("Data/middlesize.txt", TIME);
    //RNNWordLevelDataset<float>* rnnWordData = new RNNWordLevelDataset<float>("Data/shakespeare.txt", TIME);

    rnnWordData->BuildVocab();

    std::cout<<"vocab size : "<<rnnWordData->GetNumberofVocabs()<<'\n';
    std::cout<<"word number : "<<rnnWordData->GetNumberofWords()<<'\n';

    DataLoader<float> * train_dataloader = new DataLoader<float>(rnnWordData, BATCH, TRUE, 20, FALSE);

    int vocab_size = rnnWordData->GetNumberofVocabs();

    //Tensorholder<float> *x_holder = new Tensorholder<float>(Text_length, BATCH, 1, 1, vocab_size, "x");               //그냥 RNN할 때 이걸로함!
    Tensorholder<float> *x_holder = new Tensorholder<float>(TIME, BATCH, 1, 1, 1, "x");               //이거 char generation할때 이걸로
    Tensorholder<float> *label_holder = new Tensorholder<float>(TIME, BATCH, 1, 1, vocab_size, "label");


    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder, vocab_size, EMBEDDIM);
    //NeuralNetwork<float> *net = new my_SeqToSeq(x_holder, x_holder, label_holder, vocab_size);


#ifdef __CUDNN__
    std::cout<<"GPU환경에서 실행중 입니다."<<'\n';
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    // Tensor<float> *x = dataset->GetInputData();
    // Tensor<float> *label = dataset->GetLabelData();

// #ifdef __CUDNN__
//             x->SetDeviceGPU(GPUID);
//             label->SetDeviceGPU(GPUID);
// #endif  // __CUDNN__

    //std::cout<<"입력 tensor값"<<'\n'<<x<<'\n';
    //std::cout<<"-----Label 값-----"<<'\n'<<label<<'\n';

    std::cout<<'\n';
    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

    train_dataloader->StartProcess();
    // net->FeedInputTensor(2, x, label);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        //loss 값 확인할려고 내가 추가한 변수
        float temp_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        //net->FeedInputTensor(2, x_tensor, label_tensor);                        //왜??? 왜 안에 넣어두면 안되는거지???

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            std::vector<Tensor<float> *> * temp =  train_dataloader->GetDataFromGlobalBuffer();
            // printf("%d\r\n", temp->size());

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *l_t = (*temp)[1];
            delete temp;

           // std::cout<<"input"<<'\n';
           // std::cout<<x_t->GetShape()<<'\n'<<x_t<<'\n';
           //
           //  // //
           //  std::cout<<"label"<<'\n';
           //  std::cout<<l_t->GetShape()<<'\n';
           //  std::cout<<l_t<<'\n';

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

/*
            //입력 값을 잘 만들어 주는지 확인!!
            std::cout<<'\n'<<i<<"번째 입력의 값"<<'\n';
            std::cout<<x_t->GetShape()<<'\n';
            //std::cout<<x_t<<'\n';

            // int capacity = x_t->GetCapacity();
            // for(int i=0; i<capacity; i++)
            //     std::cout<<dataset->index2char((*x_t)[i]);

            //입력이 제대로 들어가는지 확인하기!!!
            for (int ba = 0; ba < BATCH; ba++) {
                for (int ti = 0; ti < TIME; ti++) {
                    std::cout<<dataset->index2char( (*x_t)[Index5D(x_t->GetShape(), ti, ba, 0, 0, 0)] );
                }
            }


            std::cout<<'\n'<<i<<"번째 Label의 값"<<'\n';
            std::cout<<l_t->GetShape()<<'\n';
            //std::cout<<l_t<<'\n';
            for (int ba = 0; ba < BATCH; ba++) {
                for (int ti = 0; ti < TIME; ti++) {
                    //maxIndex를 해줘야됨!
                    int pred_index = net->GetMaxIndex(l_t, ba, ti, vocab_size);
                    std::cout<<dataset->index2char(pred_index);
                }
            }
            //마지막에 EOS를 의미하는 E가 안보이는 이유는!!! 이제... 아래에서 출력전에 \r를 해주기 때문!
*/



            net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            //net->BPTT(Text_length);
            net->BPTT(TIME);

            //batch로 했을 경우
            //net->BPTT(time_size);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);                               // default로는 10으로 되어있음   이게 기존꺼임
            //train_avg_loss += net->GetLoss();


            train_accuracy = net->GetAccuracy(vocab_size);
            train_avg_loss = net->GetLoss();

            //std::cout<<'\n';

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),                              //+=이니깐 j+1로 나눠주는거는 알겠는데........ 근데 왜 출력되는 값이 계속 작아지는 거지??? loss값이 같아도 왜 이건 작아지는거냐고...
                   train_accuracy  /// (j + 1)
                 );
            std::cout<<'\n';

            //이것도 char 단위!!!
             // if( j%300 == 0)
             //   net->GetCharResult(dataset->GetVocab());
             fflush(stdout);

        }

        //net->GetCharResult(dataset->GetVocab());

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
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

            //새로운 textdataset--------------------------------------------------------------------------------------
            map<int, string>* Index2Vocab = rnnWordData->GetpIndex2Vocab();
            map<string, int>* Vocab2Index = rnnWordData->GetpVocab2Index();

             // std::cout<<Vocab2Index->at("than");
            // //sicinius


            //CharGenerate(int timesize, char* vocab, char startWord, int numOfClass)
            std::cout<<'\n'<<"first로 시작"<<'\n';
            startIndex = Vocab2Index->at("first");
            //startIndex = dataset->word2index("Before");
            net->GenerateSentence(TIME, Index2Vocab, startIndex, vocab_size);

            //-----------------------------------------------------------------------------------------------------

            // std::cout<<'\n'<<"menenius 시작"<<'\n';
            // startIndex = Vocab2Index->at("menenius");
            // //startIndex = dataset->word2index("Before");
            // net->GenerateSentence(TIME, Index2Vocab, startIndex, vocab_size);
            //
            //
            // std::cout<<'\n'<<"than 시작"<<'\n';
            // startIndex = Vocab2Index->at("than");
            // //startIndex = dataset->word2index("Before");
            // net->GenerateSentence(TIME, Index2Vocab, startIndex, vocab_size);





            // //char단위 - char2index  word단위 - word2index
            // //CharGenerate(int timesize, char* vocab, char startWord, int numOfClass)
            // std::cout<<'\n'<<"F로 시작"<<'\n';
            // startIndex = dataset->char2index('F');
            // //startIndex = dataset->word2index("Before");
            // net->GenerateSentence(TIME, dataset->GetVocab(), startIndex, vocab_size);
            //
            // std::cout<<'\n'<<'\n'<<"r로 시작"<<'\n';
            // startIndex = dataset->char2index('r');
            // //startIndex = dataset->word2index("First");
            // net->GenerateSentence(TIME, dataset->GetVocab(), startIndex, vocab_size);
            //
            // std::cout<<'\n'<<'\n'<<"o로 시작"<<'\n';
            // startIndex = dataset->char2index('o');
            // net->GenerateSentence(TIME, dataset->GetVocab(), startIndex, vocab_size);
            //
            // std::cout<<'\n'<<'\n'<<"m로 시작"<<'\n';
            // startIndex = dataset->char2index('m');
            // net->GenerateSentence(TIME, dataset->GetVocab(), startIndex, vocab_size);
            //
            // std::cout<<'\n'<<'\n'<<"c로 시작"<<'\n';
            // startIndex = dataset->char2index('c');
            // net->GenerateSentence(TIME, dataset->GetVocab(), startIndex, vocab_size);
            //
            // //fflush(stdout);
             std::cout << "\n\n";
        }


    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}
