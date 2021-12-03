#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <unistd.h>

#include "TextDataset.hpp"

#include "net/my_Transformer.hpp"

using namespace std;

#define BATCH               128
#define EPOCH               15
#define MAX_TRAIN_ITERATION 90000 / BATCH
#define MAX_TEST_ITERATION  0
// #define MAX_TRAIN_ITERATION  0
// #define MAX_TEST_ITERATION 10000 / BATCH


#define N_LAYER       1
#define WARMUP_STEPS  4000
#define EMBEDDING_DIM 384
#define HEAD          8

#define EOSTOK        2

#define GPUID 0

typedef struct {
    int epoch;
    float loss;
    float accuracy;
} trainData;

void PrintSentences(ofstream& fout, Tensor<float> *srcTensor, Tensor<float> *tgtTensor, Tensor<float> *predTensor, Vocabulary *srcVocabulary, Vocabulary *tgtVocabulary);

int main(int argc, char const *argv[]) {

    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

    trainData data[EPOCH];

    TextDataset<float> *dataset      = new ParalleledCorpusDataset<float>("Data/kor-eng_train_90000.txt", "kor", "eng");
    TextDataset<float> *test_dataset = new ParalleledCorpusDataset<float>("Data/kor-eng_test_10000.txt", "kor", "eng");
    dataset->BuildVocab(TRAIN);
    test_dataset->SetVocabs(dataset);
    test_dataset->BuildVocab(INFERENCE);

    DataLoader<float> *train_dataloader = new DataLoader<float>(dataset, BATCH, TRUE, 1, TRUE);
    DataLoader<float> *test_dataloader  = new DataLoader<float>(test_dataset, BATCH, TRUE, 1, TRUE);
    train_dataloader->StartProcess();
    test_dataloader->StartProcess();

    Vocabulary *srcVocabulary = ((ParalleledCorpusDataset<float> *)dataset)->GetSrcVocabulary();
    Vocabulary *tgtVocabulary = ((ParalleledCorpusDataset<float> *)dataset)->GetTgtVocabulary();
    int srcVocabSize = srcVocabulary->GetNumberofUniqueWords();
    int tgtVocabSize = tgtVocabulary->GetNumberofUniqueWords();
    int maxSeqLength = dataset->GetMaxSequenceLength();

    printf("Src Vocab 개수 : %d, Tgt Vocab 개수 : %d, Max Seq Length : %d\n", srcVocabSize, tgtVocabSize, maxSeqLength);
    
    int time_size = maxSeqLength;

    Tensorholder<float> *x_holder      = new Tensorholder<float>(1, BATCH, 1, 1, time_size, "src");
    Tensorholder<float> *dInput_holder = new Tensorholder<float>(1, BATCH, 1, 1, time_size, "tgt");
    Tensorholder<float> *label_holder  = new Tensorholder<float>(1, BATCH, time_size, 1, tgtVocabSize, "label");
    int label_shape[5] = {1, BATCH, time_size, 1, tgtVocabSize};

    NeuralNetwork<float> *net = new my_Transformer<float>(x_holder, dInput_holder, label_holder, time_size, EMBEDDING_DIM, srcVocabSize, tgtVocabSize, HEAD, N_LAYER, EOSTOK, DecoderMode::PARALLEL);

    string directory = "Parameters/";
    directory += "_" + to_string(srcVocabSize) + "_" + to_string(tgtVocabSize) + "_" + to_string(N_LAYER) + "_" + to_string(WARMUP_STEPS) + "_" + to_string(EMBEDDING_DIM) + "_" + to_string(HEAD) + "/";

    if (access(directory.c_str(), 00) == -1) {
        if (mkdir(directory.c_str(), 0766) == -1) {
            printf("mkdir fail\n");
            exit(1);
        }
    }

    string parameter_file_name = directory + "Parameter";

    cout << "<<<<<<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>>>>" << '\n';
    Container<Operator<float> *> *params     = net->GetParameter();
    int                           numOfParam = params->GetSize();

    for (int param = 0; param < numOfParam; param++) {
        std::cout << (*params)[param]->GetName() << " " << (*params)[param]->GetResult()->GetShape();
        int dimArray[5]= {0,};
        int dimCount = 0;
        for (int i = 0; i < 5; i++) {
            dimArray[i] = (*params)[param]->GetResult()->GetShape()->GetDim(i);
            if (dimArray[i] > 1) dimCount ++;
        }
        if (dimCount > 1) {
            int numInput = dimArray[3];
            int numOutput = dimArray[4];
            float v = sqrt(6.f/(numInput + numOutput));
            printf("\tRange [%f, %f]", -v, v);
            Tensor<float> *xavier_init = Tensor<float>::Random_Uniform(new Shape((*params)[param]->GetResult()->GetShape()), -v, v);
            (*params)[param]->SetResult(xavier_init);
        }
        cout << '\n';
    }

    cout << "Parameter File Name: " << parameter_file_name << '\n';

    FILE *pretrained = fopen(parameter_file_name.c_str(), "r");
    if (pretrained != NULL) {
        net->Load(pretrained);
        fclose(pretrained);
    }
    else {
        cout << "There is no Pretrained Parameters" << std::endl;
    }


#ifdef __CUDNN__
    std::cout << "GPU환경으로 변환 중입니다...." << '\t';
    net->SetDeviceGPU(GPUID);
    // testnet->SetDeviceGPU(GPUID);
    std::cout << "GPU환경에서 실행 중입니다." << '\n';
#endif

    std::cout << '\n';
    net->PrintGraphInformation();
    std::cout << "End Print Graph Information" << '\n';

    ofstream testout;
    if (MAX_TRAIN_ITERATION > 0) {
        testout.open(directory + "Test.txt");
        if (!testout.is_open()) {
            cout << "Cannot Open Test stream" << '\n';
            exit(1);
        }
    }
    else {
        testout.open(directory + "OnlyTest.txt");
        if (!testout.is_open()) {
            cout << "Cannot Open Test stream" << '\n';
            exit(1);
        }
    }   

    float best_acc = 0;
    int   epoch    = 0, step = 0;
    for (int i = epoch + 1; i <= EPOCH; i++) {

        float train_accuracy = 0.f, train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        // ============================== Train ==============================
        std::cout << "Start Train" << '\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            float newLearningRate = 2 * pow(EMBEDDING_DIM, -0.5) * min(pow(step, -0.5), step * pow(WARMUP_STEPS, -1.5));
            net->GetOptimizer()->SetLearningRate(newLearningRate);
            
            std::vector<Tensor<float> *> *temp = train_dataloader->GetDataFromGlobalBuffer();            

            Tensor<float> *x_t = (*temp)[0];
            Tensor<float> *d_t = (*temp)[1];
            Tensor<float> *l_t = (*temp)[2];
            l_t->ReShape(label_shape[0], label_shape[1], label_shape[2], label_shape[3], label_shape[4]);
#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            d_t->SetDeviceGPU(GPUID);
            l_t->SetDeviceGPU(GPUID);
#endif
            clock_t tStart = clock();

            net->FeedInputTensor(3, x_t, d_t, l_t);
            net->ResetParameterGradient();
            net->Train();

            clock_t tEnd = clock();

            train_accuracy += net->GetAccuracyWithoutPadding(tgtVocabSize, 0);
            train_avg_loss += net->GetLoss();

            if(j == MAX_TRAIN_ITERATION-1) {
                data[i-1].epoch = i;
                data[i-1].loss = train_avg_loss/(j+1);
                data[i-1].accuracy = train_accuracy/(j+1);
            }

            printf("\rTrain complete percentage is %d / %d (%d/%d) -> loss : %f, acc : %f, time : %5.2fs, time_model : %5.2fs",
                   j + 1, MAX_TRAIN_ITERATION,
                   i, EPOCH,
                   train_avg_loss/(j+1),
                   train_accuracy/(j+1),
                   (double)(clock() - tStart) / CLOCKS_PER_SEC,
                   (double)(tEnd - tStart) / CLOCKS_PER_SEC);
            delete temp;
            fflush(stdout);

            step++;
        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        if (MAX_TRAIN_ITERATION && best_acc < train_accuracy/(MAX_TRAIN_ITERATION)) {
            best_acc = train_accuracy/(MAX_TRAIN_ITERATION);
            net->Save((char *)parameter_file_name.c_str());
            cout << "Parameter Saved.. at EPOCH: " << i << '\n';
        } 

        // ======================= Test ======================
        if (MAX_TRAIN_ITERATION == 0 || train_accuracy/(MAX_TRAIN_ITERATION) > 0.95) {
            float test_accuracy = 0.f;
            float test_avg_loss = 0.f;

            net->SetModeInference();

            std::cout << "Start Test" << '\n';
            testout << "Test on " << i << '\n';
            for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

                std::vector<Tensor<float> *> *temp = test_dataloader->GetDataFromGlobalBuffer();

                Tensor<float> *x_t = (*temp)[0];
                Tensor<float> *d_t = (*temp)[1];
                Tensor<float> *l_t = (*temp)[2];
                l_t->ReShape(label_shape[0], label_shape[1], label_shape[2], label_shape[3], label_shape[4]);

    #ifdef __CUDNN__
                x_t->SetDeviceGPU(GPUID);
                d_t->SetDeviceGPU(GPUID);
                l_t->SetDeviceGPU(GPUID);
    #endif
                
                Tensor<float> *inferenced   = net->MakeInference(x_t, l_t);
                Tensor<float> *indexedLabel = l_t->Argmax(4);
                PrintSentences(testout, x_t, indexedLabel, inferenced, srcVocabulary, tgtVocabulary);

                delete inferenced;
                delete indexedLabel;
                delete x_t;
                delete d_t;
                delete l_t;
                delete temp;

                if (MAX_TRAIN_ITERATION == 0  && j % 50 == 0) {
                    Tensor<float> *src_attn         = ((my_Transformer<float> *)net)->GetDecoderSrcAttention();
                    Tensor<float> *encoder_selfattn = ((my_Transformer<float> *)net)->GetEncoderSelfAttention();
                    Tensor<float> *decoder_selfattn = ((my_Transformer<float> *)net)->GetDecoderSelfAttention();
                    src_attn->SaveTensor(directory + "E" + to_string(i) +"_T" + to_string(j)+"_DecoderSrc");
                    encoder_selfattn->SaveTensor(directory + "E" + to_string(i) +"_T" + to_string(j)+"_EncoderSelf");
                    decoder_selfattn->SaveTensor(directory + "E" + to_string(i) +"_T" + to_string(j)+"_DecoderSelf");
                }

            }
        }
    }

    if (MAX_TRAIN_ITERATION > 0) {
        string new_file_name = parameter_file_name + "_" + to_string(EPOCH);
        FILE *fp = fopen(new_file_name.c_str(), "w");
        if (fp == NULL) {
            std::cout << "cannot save\n";
        }
        else {
            net->Save(fp);
            fclose(fp);
            std::cout << "parameter saved complete\n";
        }

        string trainData_fileName = directory + "TrainData";
        ofstream ofile(trainData_fileName.c_str());
        if (ofile.is_open()) { 
            for (int i = 0; i < EPOCH; i++) {
                ofile << to_string(data[i].epoch) + ": loss = " + to_string(data[i].loss) + ", acc: " + to_string(data[i].accuracy) + "\n";
            }
            ofile.close();
        }
    }

    delete net;

    return 0;
}


void PrintSentences(ofstream& fout, Tensor<float> *srcTensor, Tensor<float> *tgtTensor, Tensor<float> *predTensor, Vocabulary *srcVocabulary, Vocabulary *tgtVocabulary) {
    map<int, string> *src_Index2Word = srcVocabulary->GetIndex2Word();
    map<int, string> *tgt_Index2Word = tgtVocabulary->GetIndex2Word();

    int colsize   = srcTensor->GetColSize();
    int batchsize = srcTensor->GetBatchSize();

    for (int ba = 0; ba < batchsize; ba++) {
        fout << "src:\t ";
        cout << "src:\t ";
        for (int co = 0; co < colsize; co++) {
            int idx = Index5D(srcTensor->GetShape(), 0, ba, 0, 0, co);
            fout << (*src_Index2Word)[(*srcTensor)[idx]] << ' ';
            cout << (*src_Index2Word)[(*srcTensor)[idx]] << ' ';
        }
        fout << "\ntgt:\t ";
        cout << "\ntgt:\t ";
        for (int co = 0; co < colsize; co++) {
            int idx = Index5D(tgtTensor->GetShape(), 0, ba, 0, 0, co);
            fout << (*tgt_Index2Word)[(*tgtTensor)[idx]] << ' ';
            cout << (*tgt_Index2Word)[(*tgtTensor)[idx]] << ' ';
        }
        fout << "\npred:\t ";
        cout << "\npred:\t ";
        for (int co = 0; co < colsize; co++) {
            int idx = Index5D(predTensor->GetShape(), 0, ba, 0, 0, co);
            fout << (*tgt_Index2Word)[(*predTensor)[idx]] << ' ';
            cout << (*tgt_Index2Word)[(*predTensor)[idx]] << ' ';
        }
        fout << '\n';
        cout << '\n';
    }
}
