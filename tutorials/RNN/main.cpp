#include "net/my_RNN.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "TextDataset.hpp"

using namespace std;

#define BATCH                 1
#define EPOCH                 2
#define MAX_TRAIN_ITERATION    20000   // (60000 / BATCH)
#define MAX_TEST_ITERATION     1   // (10000 / BATCH)
#define GPUID                 1



int main(int argc, char const *argv[]) {


    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    TextDataset<float> *dataset = new TextDataset<float>("Data/test1.txt");

    int Text_length = dataset->GetTextLength();
    int vocab_length = dataset->GetVocabLength();

    std::cout<<"파일 길이 : "<<Text_length<<" vocab 길이 : "<<vocab_length<<'\n';


    Tensorholder<float> *x_holder = new Tensorholder<float>(Text_length, BATCH, 1, 1, vocab_length, "x");
    Tensorholder<float> *label_holder = new Tensorholder<float>(Text_length, BATCH, 1, 1, vocab_length, "label");

    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder, vocab_length);


    Tensor<float> *x = dataset->GetInputData();
    Tensor<float> *label = dataset->GetLabelData();

    std::cout<<'\n';
    net->PrintGraphInformation();



    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x, label);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {



            net->ResetParameterGradient();
            net->BPTT(Text_length);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);
            //train_avg_loss += net->GetLoss();


            train_accuracy = net->GetAccuracy(vocab_length);
            train_avg_loss = net->GetLoss();


            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss,
                   train_accuracy
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

            net->BPTT_Test(Text_length);

            test_accuracy += net->GetAccuracy(vocab_length);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }

    delete net;

    return 0;
}
