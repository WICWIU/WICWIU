#include <ctime>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <unistd.h>

#include <Dataset.hpp>
#include <Utils.hpp>

#include "../../WICWIU_src/KNearestNeighbor.hpp"
#include "LFWDataset.hpp"
#include "LFWSampler.hpp"
#include "net/my_Facenet.hpp"

#define NUMBER_OF_CLASS 5749 // for lfw_funneled
//#define NUMBER_OF_CLASS               143         // for lfw_little
#define BATCH 45
//#define EPOCH                         500
#define EPOCH 10000
//#define GPUID                         0
#define GPUID 1
#define LOG_LENGTH 1
#define LEARNING_RATE_DECAY_RATE 0.1
#define LEARNING_RATE_DECAY_TIMING 100
//#define SAMPLE_PER_CLASS              5
#define SAMPLE_PER_CLASS 3
//#define KNN_K                         3
#define KNN_K 1
#define BLOCK_SIZE 360 // block to find positive and negative samples
// #define IMAGE_SIZE                   48400      // defined in LFWDataset.hpp (220*220)
// #define INPUT_DIM                   145200      // defined in LFWDataset.hpp (3 * 220 * 200)
#define FEATURE_DIM 128

#define ENABLE_TRAINING
#define ENABLE_TEST

void GetReferenceSamples(LFWDataset<float>* dataset, int dim, int noClass, int samplePerClass,
                         std::vector<float*>& vSamples, std::vector<int>& vLabels,
                         std::vector<int>* pvRefIndex = NULL,
                         int startIdx = 0); // designed for k-NN
void FindPostiveNegativeSamples(NeuralNetwork<float>* pNN, int inDim, Dataset<float>& dataset,
                                int outDim, int blockSize, int batchSize, int* pPosIndex,
                                int* pNegIndex);
void ExtractFeature(NeuralNetwork<float>* pNN, int inDim, Dataset<float>& dataset,
                    std::vector<int>& shuffle, int from, int to, int outDim, float* pFeature[],
                    int batchSize);

int main(int argc, char const* argv[])
{
    srand(time(NULL));
    time_t startTime = 0;
    struct tm* curr_tm = NULL;
    double nProcessExcuteTime = 0.;

    char filename[] = "another_params";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float>* x = new Tensorholder<float>(1, BATCH, 1, 1, 145200, "x");
    Tensorholder<float>* label = new Tensorholder<float>(1, BATCH, 1, 1, NUMBER_OF_CLASS, "label");

    // ======================= Select net ===================
    NeuralNetwork<float>* net = new my_FaceNet<float>(x, label, NUMBER_OF_CLASS);
    // net->PrintGraphInformation();

    if (access(filename, 00) == 0)
    {
        net->Load(filename);
        printf("Parameters loaded from %s\n", filename);
        fflush(stdout);
    }

#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif // __CUDNN__

    // std::cout << "/* Prepare Data */" << '\n';
    // ======================= Prepare Data ===================
    vision::Compose* transform =
        new vision::Compose({new vision::Resize(224), new vision::CenterCrop(220)});
    //    LFWDataset<float> *train_dataset = new LFWDataset<float>("./data", "lfw_little",
    //    NUMBER_OF_CLASS, transform);
    LFWDataset<float>* train_dataset =
        new LFWDataset<float>("./data", "lfw_funneled", NUMBER_OF_CLASS, transform);

#ifdef __DEBUG__
    LogMessageF("lfw_funneled_label.txt", TRUE, "%d samples\n", train_dataset->GetLength());
    for (int i = 0; i < train_dataset->GetLength(); i++)
        LogMessageF("lfw_funneled_label.txt", FALSE, "%d\t%d\n", i, train_dataset->GetLabel(i));

    for (int i = 0; i < NUMBER_OF_CLASS; i++)
    {
        printf("count[%d] = %d\n", i, train_dataset->GetSampleCount(i));
        LogMessageF("lfw_funneled_count.txt", FALSE, "count[%d] = %d\n", i,
                    train_dataset->GetSampleCount(i));
    }
    // MyPause(__FUNCTION__);
#endif //  __DEBUG__

    DataLoader<float>* train_dataloader =
        new LFWSampler<float>(NUMBER_OF_CLASS, train_dataset, BATCH, TRUE, 1, FALSE);
    std::cout << "test" << '\n';

#ifdef ENABLE_TEST
    // get referece idxs for k-NN
    std::vector<float*> vRefSamples;
    std::vector<int> vRefLabels;
    std::vector<int> vPosIndex;
    std::vector<int> vNegIndex;
    std::cout << "test1" << '\n';
    GetReferenceSamples(train_dataset, INPUT_DIM, NUMBER_OF_CLASS, SAMPLE_PER_CLASS, vRefSamples,
                        vRefLabels);
#endif // ENABLE_TEST
    std::cout << "test2" << '\n';
    //    LFWDataset<float> *test_dataset = new LFWDataset<float>("./data", "lfw_Test", 60,
    //    transform);

    float best_acc = 0.f;
    int startEpoch = -1; // epoch starts from startEpoch + 1
                         //    int   startEpoch = 2614;

    int loop_for_train = (train_dataset->GetLength() - train_dataset->GetNoMinorClass(2)) / BATCH;
    float min_lr = 0.000001F;
    int pos_neg_cycle = 3;

    std::cout << "filename : " << filename << '\n';
    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "start epoch : " << startEpoch << '\n';
    std::cout << "end epoch : " << EPOCH << '\n';
    std::cout << "min_lr : " << min_lr << '\n';
    std::cout << "pos_neg_cycle : " << pos_neg_cycle << '\n';

    if (startEpoch / LEARNING_RATE_DECAY_TIMING)
    {
        float lr = net->GetOptimizer()->GetLearningRate();
        float new_lr = lr * pow(0.1, (int)(startEpoch / LEARNING_RATE_DECAY_TIMING));
        if (new_lr < min_lr)
            new_lr = min_lr;
        net->GetOptimizer()->SetLearningRate(new_lr);
        std::cout << "lr : " << new_lr << '\n';
    }

    for (int epoch = startEpoch + 1; epoch < EPOCH; epoch++)
    {
        std::cout << "epoch : " << epoch << '\n';

        float train_avg_loss = 0.f;
        float train_cur_loss = 0.f;

#ifdef ENABLE_TRAINING
        if (epoch == startEpoch + 1 || (epoch - 1) % pos_neg_cycle == 0)
        {
            printf("Searching for positive and negative samples...\n");
            fflush(stdout);
            LogMessageF("log.txt", FALSE, "Finding positive and negative samples...\n");
            train_dataset->GetPositiveIndices().resize(train_dataset->GetLength());
            train_dataset->GetNegativeIndices().resize(train_dataset->GetLength());

            FindPostiveNegativeSamples(net, INPUT_DIM, *train_dataset, FEATURE_DIM, BLOCK_SIZE,
                                       BATCH, &train_dataset->GetPositiveIndices()[0],
                                       &train_dataset->GetNegativeIndices()[0]);

            printf("Searching for positive and negative samples... Done.\n");
            fflush(stdout);

#ifdef __DEBUG__
            for (int i = 0; i < 100; i++)
                printf("%d (%s): anchor: %d, pos:%d, neg: %d\n", i,
                       train_dataset->GetImagePath(i).c_str(), train_dataset->GetLabel(i),
                       train_dataset->GetPositiveIndex(i), train_dataset->GetNegativeIndex(i));
                // MyPause(__FUNCTION__);
#endif // __DEBUG__
        }

        startTime = time(NULL);
        curr_tm = localtime(&startTime);
        std::cout << curr_tm->tm_hour << "\'h " << curr_tm->tm_min << "\'m " << curr_tm->tm_sec
                  << "\'s" << '\n';

        float lr = net->GetOptimizer()->GetLearningRate();
        if (epoch % LEARNING_RATE_DECAY_TIMING == 0 && lr > min_lr)
        {
            // adjust learning rate
            std::cout << "Change learning rate!" << '\n';
            float new_lr = lr * LEARNING_RATE_DECAY_RATE;
            if (new_lr < min_lr)
                new_lr = min_lr;
            net->GetOptimizer()->SetLearningRate(new_lr);
            std::cout << "lr : " << new_lr << '\n';
        }
        else
        {
            std::cout << "lr : " << lr << '\n';
        }

        // ======================= Train =======================
        net->SetModeTrain();

        for (int j = 0; j < loop_for_train; j++)
        {
            std::vector<Tensor<float>*>* temp = train_dataloader->GetDataFromGlobalBuffer();

#ifdef __CUDNN__
            (*temp)[0]->SetDeviceGPU(GPUID); // 異뷀썑 ?먮룞???꾩슂
            (*temp)[1]->SetDeviceGPU(GPUID);
#endif // __CUDNN__

            net->FeedInputTensor(2, (*temp)[0], (*temp)[1]);

            delete temp;
            temp = NULL;

            net->ResetParameterGradient();
            net->Train();

            train_cur_loss = net->GetLoss();
            train_avg_loss += train_cur_loss;

            printf("\r%d / %d -> cur_loss : %f", j + 1, loop_for_train, train_avg_loss / (j + 1));
            fflush(stdout);
        }
        printf("\n");

#endif // ENABLE_TRAINING

        // ======================= Test ======================
        float test_accuracy = 0.f;
        // float test_avg_loss = 0.f;

#ifdef ENABLE_TEST
        net->SetModeInference();

        printf("Start testing...\n");
        fflush(stdout);
        //        LFWDataset<float> &dataset = *test_dataset;
        LFWDataset<float>& dataset = *train_dataset; // only for debug

        // create k-NN classifier using net and (vRefLabels, vRefSamples)
        // printf("Feature fxtraction (reference images)\n");  fflush(stdout);
        std::vector<float*> vRefFeatures;
        AllocFeatureVector(FEATURE_DIM, vRefSamples.size(), vRefFeatures);
        net->InputToFeature(INPUT_DIM, vRefSamples.size(), &vRefSamples[0], FEATURE_DIM,
                            &vRefFeatures[0],
                            BATCH); // convert reference images to feature vectors using CNN
        KNearestNeighbor knn(FEATURE_DIM, NUMBER_OF_CLASS, vRefSamples.size(), &vRefLabels[0],
                             &vRefFeatures[0]); // create k-NN classifier
        DeleteFeatureVector(vRefFeatures);

        // test
        int correct = 0;
        int noTestSample = dataset.GetLength();
        int noBatch = (noTestSample + BATCH - 1) / BATCH;
        int remained = noTestSample;

        std::vector<float*> vTestSample;
        std::vector<float*> vTestFeature;

        vPosIndex.resize(dataset.GetLength());
        vNegIndex.resize(dataset.GetLength());

        printf("Recognizing using knn...\n");
        fflush(stdout);
        for (int batchIdx = 0; batchIdx < noBatch; batchIdx++)
        {
            int curBatch = MIN(remained, BATCH);

            // extract feature using CNN
            AllocFeatureVector(INPUT_DIM, curBatch, vTestSample);
            AllocFeatureVector(FEATURE_DIM, curBatch, vTestFeature);

            // printf("\rReading test images (batchIdx = %d)... (%s %d)\n", batchIdx, __FILE__,
            // __LINE__);  fflush(stdout);
            for (int i = 0; i < curBatch; i++)
            {
                //                printf("Reading test image (i = %d)... (%s %d)\n", i, __FILE__,
                //                __LINE__);  fflush(stdout);
                dataset.CopyData(batchIdx * BATCH + i, vTestSample[i]);
            }

            // printf("Extracting feature ... (%s %d)\n", __FILE__, __LINE__);  fflush(stdout);

            net->InputToFeature(INPUT_DIM, vTestSample.size(), &vTestSample[0], FEATURE_DIM,
                                &vTestFeature[0], BATCH);

            // printf("Recognizing test images ... (%s %d)\n", __FILE__, __LINE__);  fflush(stdout);
            // recognize using k-NN classifier
            for (int i = 0; i < curBatch; i++)
            {
                int result = knn.Recognize(vTestFeature[i], KNN_K);
                if (result == dataset.GetLabel(batchIdx * BATCH + i))
                    correct++;
            }

            DeleteFeatureVector(vTestFeature);
            DeleteFeatureVector(vTestSample);

            remained -= curBatch;

            if ((batchIdx + 1) % 10 == 0)
            {
                printf("batch = %d / %d test accuracy = %f (%d / %d)\n", batchIdx + 1, noBatch,
                       correct / (float)(batchIdx * BATCH + curBatch), correct,
                       batchIdx * BATCH + curBatch);
                fflush(stdout);
            }
        }

        test_accuracy = correct / (float)noTestSample;
        printf("\repoch = %d, test accuracy = %f (%d / %d)\n", epoch, correct / (float)noTestSample,
               correct, noTestSample);
        fflush(stdout);
        LogMessageF("log.txt", FALSE,
                    "epoch = %d, lr = %f,  training loss = %f, test accuracy = %f (%d / %d)\n",
                    epoch, net->GetOptimizer()->GetLearningRate(),
                    train_avg_loss / (float)loop_for_train, correct / (float)noTestSample, correct,
                    noTestSample);

        if (test_accuracy > best_acc)
        {
            best_acc = test_accuracy;
            printf("Saving best model into %s (best_acc = %f)\n", filename, best_acc);
            net->Save(filename);
        }

#ifndef ENABLE_TRAINING
        break; // if training is disabled, repeating test is meaningless
#endif         // ENABLE_TRAINING

#endif // ENABLE_TEST

        if (test_accuracy > best_acc)
        {
            // save the best model
            best_acc = test_accuracy;
            net->Save(filename);

            printf("Model was saved in %s. (best_acc = %f)\n", filename, best_acc);
            fflush(stdout);
        }

        printf("\n");
    }

    delete net;
    delete train_dataloader;
    delete train_dataset;
    //    delete test_dataset;

    return 0;
}

void GetReferenceSamples(LFWDataset<float>* dataset, int dim, int noClass, int samplePerClass,
                         std::vector<float*>& vSamples, std::vector<int>& vLabels,
                         std::vector<int>* pvRefIndex, int startIdx) // designed for k-NN
{
    DeleteFeatureVector(vSamples);
    vLabels.resize(0);
    if (pvRefIndex != NULL)
        pvRefIndex->resize(0);

    if (startIdx = -1)
        startIdx = rand() % dataset->GetLength();

    std::vector<int> vCount;
    vCount.resize(noClass);
    for (int i = 0; i < noClass; i++)
        vCount[i] = 0;

    int noComplete = 0;
    for (int i = 0; i < dataset->GetLength() && noComplete < noClass; i++)
    {
        int idx = (startIdx + i) % dataset->GetLength();
        int label = dataset->GetLabel(idx);
        if (label >= noClass)
        {
            printf("Error! label = %d, noClass = %d\n", label, noClass);
            printf("Press Enter to continue (%s)...", __FUNCTION__);
            fflush(stdout);
            getchar();
        }
        if (vCount[label] < samplePerClass)
        {
            vLabels.push_back(label);
            if (pvRefIndex)
                pvRefIndex->push_back(idx);

            float* pSample = new float[dim];
            if (pSample == NULL)
            {
                printf("Failed to allocate memory, dim = %d in %s (%s %d)\n", dim, __FUNCTION__,
                       __FILE__, __LINE__);
                exit(-1);
            }

            dataset->CopyData(idx, pSample);
            vSamples.push_back(pSample);

            vCount[label]++;
            if (vCount[label] == samplePerClass)
                noComplete++;
        }
    }

#ifdef __DEBUG__
    for (int i = 0; i < vLabels.size(); i++)
    {
        printf("vLabels[%d] = %d\n", i, vLabels[i]);
    }
#endif // __DEBUG__
}

void FindPostiveNegativeSamples(NeuralNetwork<float>* pNN, int inDim, Dataset<float>& dataset,
                                int outDim, int blockSize, int batchSize, int* pPosIndex,
                                int* pNegIndex)
{
    int noSample = dataset.GetLength();
    int noBlock = (noSample + blockSize - 1) / blockSize;

    std::vector<float*> vBlockFeature;
    AllocFeatureVector(outDim, blockSize, vBlockFeature);

    std::vector<int> shuffle;
    std::vector<float> vMinNegative;
    std::vector<float> vMaxPositive;
    try
    {
        shuffle.resize(noSample);
        vMinNegative.resize(blockSize);
        vMaxPositive.resize(blockSize);
    }
    catch (...)
    {
        printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return;
    }

    for (int i = 0; i < noSample; i++)
    {
        shuffle[i] = i;
        pPosIndex[i] = -1;
        pNegIndex[i] = -1;
    }

    // random shuffle
    // for(int i = 0; i < noSample; i++){
    //     int j = rand() % noSample;
    //     int temp = shuffle[i];
    //     shuffle[i] = shuffle[j];
    //     shuffle[j] = temp;
    // }
    std::random_shuffle(shuffle.begin(), shuffle.end());

    int remained = noSample;
    for (int block = 0; block < noBlock; block++)
    {
        int from = block * blockSize;

        if (block % 5 == 0)
        {
            printf("\rProcessing block %d / %d\n", block, noBlock);
            fflush(stdout);
        }

        int curBlock = MIN(blockSize, remained);
        int to = from + curBlock;

        //      printf("Extracting feature\n");       fflush(stdout);
        ExtractFeature(pNN, inDim, dataset, shuffle, from, to, outDim, &vBlockFeature[0],
                       batchSize);

        for (int i = 0; i < curBlock; i++)
        {
            vMinNegative[i] = FLT_MAX;
            vMaxPositive[i] = -FLT_MAX;
        }

        // printf("Finding positive and negative samples\n");       fflush(stdout);
        for (int i = 0; i < curBlock; i++)
        {
            int idx_i = shuffle[from + i];
            for (int j = i + 1; j < curBlock; j++)
            {
                int idx_j = shuffle[from + j];

                int dist2 = GetSquareDistance(outDim, vBlockFeature[i], vBlockFeature[j]);

                if (dataset.GetLabel(idx_i) == dataset.GetLabel(idx_j))
                { // same label
                    if (dist2 > vMaxPositive[i])
                    {
                        pPosIndex[idx_i] = idx_j;
                        vMaxPositive[i] = dist2;
                    }
                    if (dist2 > vMaxPositive[j])
                    {
                        pPosIndex[idx_j] = idx_i;
                        vMaxPositive[j] = dist2;
                    }

                    if (dataset.GetLabel(idx_i) != dataset.GetLabel(idx_j))
                    {
                        printf("Error! Wrong positive sample! (%s %d)\n", __FILE__, __LINE__);
                        MyPause(__FUNCTION__);
                    }
                }
                else
                { // different label
                    if (dist2 < vMinNegative[i])
                    {
                        pNegIndex[idx_i] = idx_j;
                        vMinNegative[i] = dist2;
                    }
                    if (dist2 < vMinNegative[j])
                    {
                        pNegIndex[idx_j] = idx_i;
                        vMinNegative[j] = dist2;
                    }

                    if (dataset.GetLabel(idx_i) == dataset.GetLabel(idx_j))
                    {
                        printf("Error! Wrong negative sample! (%s %d)\n", __FILE__, __LINE__);
                        MyPause(__FUNCTION__);
                    }
                }
            }
        }

#ifdef __DEBUG__
        for (int i = 0; i < curBlock; i++)
        {
            int anchorIdx = shuffle[from + i];
            int posIdx = pPosIndex[anchorIdx];
            int negIdx = pNegIndex[anchorIdx];

            int posLabel = (posIdx >= 0 ? dataset.GetLabel(posIdx) : -1);
            int negLabel = (negIdx >= 0 ? dataset.GetLabel(negIdx) : -1);
            printf("%d: anchor: %d (%d),\tpos = %d (%d),\tneg = %d (%d)\n", i, anchorIdx,
                   dataset.GetLabel(anchorIdx), posIdx, posLabel, negIdx, negLabel);

            // if(dataset.GetLabel(anchorIdx) != dataset.GetLabel(posIdx) ||
            // dataset.GetLabel(anchorIdx) == dataset.GetLabel(negIdx)){
            //     printf("Error! Wrong positive or negative sample! (%s %d)\n", __FILE__,
            //     __LINE__); MyPause(__FUNCTION__);
            // }
        }
        MyPause(__FUNCTION__);
#endif //  __DEBUG__

        remained -= blockSize;
        // printf("Done\n");       fflush(stdout);
    }

    DeleteFeatureVector(vBlockFeature);

    printf("Done\n");
    fflush(stdout);
}

void ExtractFeature(NeuralNetwork<float>* pNN, int inDim, Dataset<float>& dataset,
                    std::vector<int>& shuffle, int from, int to, int outDim, float* pFeature[],
                    int batchSize)
{
    from = MIN(from, dataset.GetLength());
    to = MIN(to, dataset.GetLength());
    if (from == to)
    {
        printf("No data to process in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return;
    }

    int noSample = to - from;
    batchSize = MIN(batchSize, noSample);
    int noBatch = (noSample + batchSize - 1) / batchSize;
    int remained = noSample;

    // allocate temporary buffers
    std::vector<float*> vTmpSample;
    std::vector<float*> vTmpFeature;
    try
    {
        AllocFeatureVector(inDim, batchSize, vTmpSample);
        AllocFeatureVector(outDim, batchSize, vTmpFeature);
    }
    catch (...)
    {
        printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        MyPause(__FUNCTION__);
        return;
    }

    // extract feature using CNN
    for (int batchIdx = 0; batchIdx < noBatch; batchIdx++)
    {
        int curBatch = MIN(remained, batchSize);

        // printf("Reading batch %d\n", batchIdx);     fflush(stdout);
        for (int i = 0; i < curBatch; i++)
            dataset.CopyData(shuffle[from + batchIdx * batchSize + i], vTmpSample[i]);

        // printf("Calling InputToFeature()");     fflush(stdout);
        AllocFeatureVector(outDim, vTmpSample.size(), vTmpFeature);
        pNN->InputToFeature(inDim, vTmpSample.size(), &vTmpSample[0], outDim, &vTmpFeature[0],
                            batchSize);

        // printf("Copying feature");     fflush(stdout);
        for (int i = 0; i < curBatch; i++)
            //            memcpy(pFeature[batchIdx * batchSize + i], vTmpFeature[i], outDim *
            //            sizeof(vTmpFeature[i][0]));
            memcpy(pFeature[batchIdx * batchSize + i], vTmpFeature[i], outDim * sizeof(float));

        DeleteFeatureVector(vTmpFeature);
        // printf("Done.");     fflush(stdout);

        remained -= batchSize;
    }

    DeleteFeatureVector(vTmpSample);
}
