
#include "net/my_SeqToSeq.hpp"
#include "net/my_AttentionSeqToSeq.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더
#include <map>
#include "TextDataset.hpp"

using namespace std;

#define EMBEDDIM               350            // 이걸 늘리면 segfault가 발생하는데... 이상하네....
// #define ENCODER_TIME           3
// #define DECODER_TIME           4
#define BATCH                  64
#define EPOCH                  10
#define MAX_TRAIN_ITERATION    1500   // (60000 / BATCH)
#define MAX_TEST_ITERATION     5   // (10000 / BATCH)
#define GPUID                  6



//이거는 toy example을 위한 main 함수!
//      Get a job.	Trouve un emploi !


int main(int argc, char const *argv[]) {


    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    //RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/eng-fra_short.txt", "eng", "fra");      //input 2개는 확인해 봤지만 label은 확인하지 못함!
    RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/test3.txt", "eng", "fra");
    //RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/test2.txt", "eng", "fra");
    //RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/padding_test.txt", "eng", "fra");
    translation_data->BuildVocab();

    DataLoader<float> * train_dataloader = new DataLoader<float>(translation_data, BATCH, TRUE, 20, FALSE);

    int EncoderTime = translation_data->GetEncoderMaxTime();
    int DecoderTime = translation_data->GetDecoderMaxTime();
    int vocab_size  = translation_data->GetNumberofVocabs();

    std::cout<<"------------------------------------------------"<<'\n';
    std::cout<<"Encoder Time : "<<EncoderTime<<'\n'<<"Decoder Time : "<<DecoderTime<<'\n'<<"vocab_size : "<<vocab_size<<'\n';

    //Dataloader를 사용해서 하면 마지막 col 부분은 1로! 단 label은 cross entropy를 사용해서 one-hot으로 넘겨줘야 되니깐 vocab_size로 해야됨
    Tensorholder<float> *encoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(EncoderTime, BATCH, 1, 1, 1), "Encoder_input");
    Tensorholder<float> *decoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, 1), "Decoder_input");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Tensor<float>::Zeros(DecoderTime, BATCH, 1, 1, vocab_size), "label");
    Tensorholder<float> *encoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "EncoderLengths");
    Tensorholder<float> *decoder_lengths_holder = new Tensorholder<float>(Tensor<float>::Zeros(1, BATCH, 1, 1, 1), "DecoderLengths");

    //NeuralNetwork<float> *net = new my_SeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, vocab_size, EMBEDDIM);
    NeuralNetwork<float> *net = new my_SeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, vocab_size, EMBEDDIM, encoder_lengths_holder, decoder_lengths_holder);
    //NeuralNetwork<float> *net = new my_AttentionSeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, encoder_lengths_holder, decoder_lengths_holder, vocab_size, EMBEDDIM);

#ifdef __CUDNN__
    std::cout<<"GPU환경에서 실행중 입니다."<<'\n';
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__


    std::cout<<'\n';
    net->PrintGraphInformation();

    //단어 전처리 부분 확인하는 코드!
     map<int, string> *index2vocab = translation_data->GetpIndex2Vocab();
    // for ( int i=0; i< translation_data->GetNumberofVocabs(); i++ ){
    //     std::cout<<i<<" : "<<index2vocab->at(i)<<'\n';
    // }

    float best_acc = 0;
    int   epoch    = 0;

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

            // std::cout<<"encoder input"<<'\n';
            // std::cout<<'\n'<<e_t->GetShape()<<'\n'<<e_t<<'\n';
            //
            //  std::cout<<"Decoder input"<<'\n';
            //  std::cout<<d_t->GetShape()<<'\n'<<d_t<<'\n';
            // std::cout<<l_t->GetShape()<<'\n';
            // //
            // std::cout<<"Deocder label"<<'\n';
            // std::cout<<l_t->GetShape()<<'\n';
            // std::cout<<l_t<<'\n';

            //   std::cout<<'\n';
            // std::cout<<"encoder length"<<'\n';
            // std::cout<<e_l<<'\n';
            //
            // std::cout<<"decoder length"<<'\n';
            // std::cout<<d_l<<'\n';

            //label은 one-hot으로 만들어서... 값의 확인이 어려워 보임....


            net->FeedInputTensor(5, e_t, d_t, l_t, e_l, d_l);
            net->ResetParameterGradient();
            //net->BPTT(Text_length);
            //net->BPTT(DecoderTime);
        #ifdef __CUDNN__
            net->seq2seqBPTTOnGPU(EncoderTime, DecoderTime);         //GPU함수가 있는가!
        #else
            net->seq2seqBPTT(EncoderTime, DecoderTime);
        #endif

            //batch로 했을 경우
            //net->BPTT(time_size);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);                               // default로는 10으로 되어있음   이게 기존꺼임
            //train_avg_loss += net->GetLoss();

            train_accuracy = net->GetAccuracy(vocab_size, d_l);
            train_avg_loss = net->GetLoss(d_l); //d_l

            //std::cout<<'\n';

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),                              //+=이니깐 j+1로 나눠주는거는 알겠는데........ 근데 왜 출력되는 값이 계속 작아지는 거지??? loss값이 같아도 왜 이건 작아지는거냐고...
                   train_accuracy  /// (j + 1)
                 );
            //std::cout<<'\n';

             fflush(stdout);

        }

        //net->GetCharResult(dataset->GetVocab());

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;          //clocks_per_sec로 나눠서 이제 단위가 초!!
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

            //std::cout<<e_t<<'\n';

            //입력 확인
            Shape* eShape = e_t->GetShape();
            for(int en = 0; en < EncoderTime; en ++){
                std::cout<<index2vocab->at((*e_t)[Index5D(eShape, en, 0, 0, 0, 0)])<<" ";
            }
            std::cout<<'\n';

            // //e_t = translation_data->GetTestData("We're not desperate yet.");
            net->FeedInputTensor(4, e_t, d_t, l_t, e_l);      //아니면 함수의 인자로 넘겨주던가....
            //

            map<int, string>* index2vocab = translation_data->GetpIndex2Vocab();
        #ifdef __CUDNN__
            net->SentenceTranslateOnGPU(index2vocab);
        #else
            net->SentenceTranslate(index2vocab);
        #endif

              std::cout << "\n\n";
        }


    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}











/*
// Dataloader 사용하기 전에 test 하기위해 작성한 main
// 즉 toy example


#include "net/my_SeqToSeq.hpp"
#include "net/my_AttentionSeqToSeq.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더
#include "TextDataset.hpp"

using namespace std;

#define EMBEDDIM               32
#define ENCODER_TIME           3
#define DECODER_TIME           4
#define BATCH                  1
#define EPOCH                  5
#define MAX_TRAIN_ITERATION    300   // (60000 / BATCH)
#define MAX_TEST_ITERATION     1   // (10000 / BATCH)
#define GPUID                  2



//이거는 toy example을 위한 main 함수!
//      Get a job.	Trouve un emploi !


int main(int argc, char const *argv[]) {


    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    //DataSet 생성!
    //ParalleledCorpusDataset<float>* translation_data = new ParalleledCorpusDataset<float>("Data/eng-fra_short.txt", "eng", "fra");
    RNNParalleledCorpusDataset<float>* translation_data = new RNNParalleledCorpusDataset<float>("Data/eng-fra_short.txt", "eng", "fra");
    translation_data->BuildVocab();

    DataLoader<float> * train_dataloader = new DataLoader<float>(translation_data, BATCH, TRUE, 20, FALSE);

    int vocab_size  = 8;

    //Dataloader를 사용해서 하면 마지막 col 부분은 1로! 단 label은 cross entropy를 사용해서 one-hot으로 넘겨줘야 되니깐 vocab_size로 해야됨
    Tensorholder<float> *encoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(ENCODER_TIME, BATCH, 1, 1, vocab_size), "Encoder_input");
    Tensorholder<float> *decoder_x_holder = new Tensorholder<float>(Tensor<float>::Zeros(DECODER_TIME, BATCH, 1, 1, vocab_size), "Decoder_input");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Tensor<float>::Zeros(DECODER_TIME, BATCH, 1, 1, vocab_size), "label");

    //NeuralNetwork<float> *net = new my_SeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, vocab_size);
    NeuralNetwork<float> *net = new my_AttentionSeqToSeq(encoder_x_holder, decoder_x_holder, label_holder, vocab_size, EMBEDDIM);

    //one-hot vector 만들기
    (*(encoder_x_holder->GetResult()))[0] = 1;
    (*(encoder_x_holder->GetResult()))[9] = 1;
    (*(encoder_x_holder->GetResult()))[18] = 1;

    (*(decoder_x_holder->GetResult()))[3] = 1;
    (*(decoder_x_holder->GetResult()))[12] = 1;
    (*(decoder_x_holder->GetResult()))[21] = 1;
    (*(decoder_x_holder->GetResult()))[30] = 1;


    (*(label_holder->GetResult()))[4] = 1;
    (*(label_holder->GetResult()))[13] = 1;
    (*(label_holder->GetResult()))[22] = 1;
    (*(label_holder->GetResult()))[31] = 1;

    //입력하고 label 값 확인해보기!

    std::cout<<"입력 확인하기"<<'\n';
    std::cout<<encoder_x_holder->GetResult()<<'\n';
    std::cout<<decoder_x_holder->GetResult()<<'\n';
    std::cout<<label_holder->GetResult()<<'\n';


    std::cout<<'\n';
    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

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



            //net->FeedInputTensor(2, x_t, l_t);
            net->ResetParameterGradient();
            //net->BPTT(Text_length);
            net->BPTT(DECODER_TIME);

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
            //std::cout<<'\n';

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

            int startIndex = 0;



            //동작하는거 같음!!!!
            //이상한 방법이지만 길이 늘려보기!!!
            // Tensor<float> *x_t = new Tensor<float>(TIME, BATCH, 1, 1, 1);
            // Tensor<float> *l_t = new Tensor<float>(TIME, BATCH, 1, 1, vocab_size);
            // net->FeedInputTensor(2, x_t, l_t);




            std::cout << "\n\n";
        }


    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}
*/
