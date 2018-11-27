#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "../../WICWIU_src/Tensor_utils.hpp"

#define NUMBER_OF_CLASS                        10
#define CAPACITY_OF_IMAGE                      3072
#define BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG    3073
#define SIZE_OF_SINGLE_DATA_FILE               30730000
#define LEGNTH_OF_WIDTH_AND_HEIGHT             32

using namespace std;

template<typename DTYPE> class CIFAR10Reader {
private:
    string m_path     = "cifar-10-batches-bin";
    string train_data = "data_batch_";
    string test_data  = "test_batch.bin";

    /*Train image*/
    // for shuffle class index
    vector<int> m_shuffledList;
    // number of img of each class
    unsigned char **m_aaTrainImageSrcs;

    /*Test image*/
    unsigned char *m_aTestImageSrc;

    // batch Tensor << before concatenate
    queue<Tensor<DTYPE> *> *m_aaSetOfImage;  // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabel;  // size : batch size

    // Storage for preprocessed Tensor
    queue<Tensor<DTYPE> **> *m_aaQForData;  // buffer Size is independently define here

    int m_batchSize;
    int m_recallnum;
    int m_bufferSize;

    int m_isTrain;

    pthread_t m_thread;

    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    int m_work;

    /*Data Augmentation*/
    // for normalization
    int m_useNormalization;
    int m_isNormalizePerChannelWise;
    float *m_aMean;
    float *m_aStddev;

    // Random_crop
    int m_useRandomCrop;
    int m_padding;
    int m_lengthOfWidthAndHeight;
    vector<int> m_shuffledListForCrop;

    // Flip
    int m_useRandomHorizontalFlip;
    vector<int> m_shuffledListForFlip;

private:
    int Alloc() {
        if (m_isTrain) {
            for (int i = 1; i < 50000; i++) m_shuffledList.push_back(i);
            m_aaTrainImageSrcs = (unsigned char **)malloc(sizeof(unsigned char *) * 5);
            // new unsigned char *[5];

            for (int i = 0; i < 5; i++) {
                char   num       = i + 49; // i to a > "i+1"
                string filePath  = m_path + '/' + train_data + num + ".bin";
                const char *cstr = filePath.c_str();

                std::cout << filePath << '\n';

                FILE *pFile = NULL;

                pFile = fopen(cstr, "rb");

                m_aaTrainImageSrcs[i] = (unsigned char *)malloc(sizeof(unsigned char) * SIZE_OF_SINGLE_DATA_FILE);

                fread(m_aaTrainImageSrcs[i], sizeof(unsigned char), SIZE_OF_SINGLE_DATA_FILE, pFile);

                fclose(pFile);
            }
        } else {
            string filePath  = m_path + '/' + test_data;
            const char *cstr = filePath.c_str();

            std::cout << filePath << '\n';

            FILE *pFile = NULL;

            pFile = fopen(cstr, "rb");

            m_aTestImageSrc = (unsigned char *)malloc(sizeof(unsigned char) * SIZE_OF_SINGLE_DATA_FILE);

            fread(m_aTestImageSrc, sizeof(unsigned char), SIZE_OF_SINGLE_DATA_FILE, pFile);

            fclose(pFile);
        }

        m_aaSetOfImage = new queue<Tensor<DTYPE> *>();  // Each tensor shows single image
        m_aaSetOfLabel = new queue<Tensor<DTYPE> *>();

        m_aaQForData = new queue<Tensor<DTYPE> **>();  // Each tensor shows set of image which size is batchSize

        return TRUE;
    }

    void Delete() {
        if (m_aaTrainImageSrcs) {
            // We cannot dealloc this part, is it charactor of string value?
            for (int i = 0; i < 5; i++) {
                if (m_aaTrainImageSrcs[i]) {
                    // std::cout << m_aaImagesOfClass[i] << '\n';
                    free(m_aaTrainImageSrcs[i]);
                    m_aaTrainImageSrcs[i] = NULL;
                }
            }
            free(m_aaTrainImageSrcs);
        }

        if (m_aTestImageSrc) {
            // std::cout << m_aaImagesOfClass[i] << '\n';
            free(m_aTestImageSrc);
            m_aTestImageSrc = NULL;
        }

        if (m_aaSetOfImage) {
            if (m_aaSetOfImage->size() != 0) {
                int numOfTensor = m_aaSetOfImage->size();

                for (int i = 0; i < numOfTensor; i++) {
                    delete m_aaSetOfImage->front();
                    m_aaSetOfImage->front() = NULL;
                    m_aaSetOfImage->pop();
                }
            }
            delete m_aaSetOfImage;
            m_aaSetOfImage = NULL;
        }

        if (m_aaSetOfLabel) {
            if (m_aaSetOfLabel->size() != 0) {
                int numOfTensor = m_aaSetOfLabel->size();

                for (int i = 0; i < numOfTensor; i++) {
                    delete m_aaSetOfLabel->front();
                    m_aaSetOfLabel->front() = NULL;
                    m_aaSetOfLabel->pop();
                }
            }
            delete m_aaSetOfLabel;
            m_aaSetOfLabel = NULL;
        }

        if (m_aaQForData) {
            if (m_aaQForData->size() != 0) {
                int numOfTensor = m_aaQForData->size();

                for (int i = 0; i < numOfTensor; i++) {
                    Tensor<DTYPE> **temp = m_aaQForData->front();
                    m_aaQForData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete[] temp;
                    temp = NULL;
                }
            }
            delete m_aaQForData;
            m_aaQForData = NULL;
        }

        if (m_useNormalization) {
            if (m_aMean) {
                delete[] m_aMean;
                m_aMean = NULL;
            }

            if (m_aStddev) {
                delete[] m_aStddev;
                m_aStddev = NULL;
            }
        }
    }

public:
    CIFAR10Reader(int batchSize, int bufferSize, int isTrain) {
        m_aaTrainImageSrcs = NULL;
        m_aTestImageSrc    = NULL;
        m_aaSetOfImage     = NULL;
        m_aaSetOfLabel     = NULL;
        m_aaQForData       = NULL;

        m_batchSize  = batchSize;
        m_isTrain    = isTrain;
        m_recallnum  = 0;
        m_bufferSize = bufferSize;

        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_work = 1;

        m_useRandomCrop          = FALSE;
        m_padding                = 0;
        m_lengthOfWidthAndHeight = LEGNTH_OF_WIDTH_AND_HEIGHT;

        m_useNormalization          = FALSE;
        m_isNormalizePerChannelWise = FALSE;
        m_aMean                     = NULL;
        m_aStddev                   = NULL;

        m_useRandomHorizontalFlip = FALSE;

        Alloc();

        // prepare data what we need
        // start data preprocessing with above information
        // It works with thread
        // it will be end when receive "STOP" signal
    }

    virtual ~CIFAR10Reader() {
        Delete();
    }

    int UseNormalization(int isNormalizePerChannelWise = TRUE, CIFAR10Reader<DTYPE> *src = NULL) {
        m_useNormalization          = TRUE;
        m_isNormalizePerChannelWise = isNormalizePerChannelWise;

        if (m_isTrain) {
            CalculateTrainDataMeanAndStddev();
        } else {
            if (src) {
                if (m_isNormalizePerChannelWise != src->GetNormalizationMode()) {
                    std::cout << "caution! m_isNormalizePerChannelWise is different between src!" << '\n';
                    m_isNormalizePerChannelWise = src->GetNormalizationMode();
                }
                float *means   = src->GetSetOfMean();
                float *stddevs = src->GetSetOfStddev();

                CopyMeanAndStddevFromSrc(means, stddevs);
            } else {
                std::cout << "their is no src as input! - CopyMeanAndStddevFromSrc" << '\n';
            }
        }

        return TRUE;
    }

    int CalculateTrainDataMeanAndStddev() {
        int imgNum = 0;  // random image of above class
        int srcNum = 0;

        unsigned char *imgSrc;
        unsigned char *ip;

        if (m_isNormalizePerChannelWise) {
            m_aMean   = new float[3];
            m_aStddev = new float[3];

            for (int channelNum = 0; channelNum < 3; channelNum++) {
                m_aMean[channelNum]   = 0.f;
                m_aStddev[channelNum] = 0.f;
            }

            float tempMean = 0.f;

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10000; j++) {
                    srcNum = i;
                    imgNum = j;

                    imgSrc = m_aaTrainImageSrcs[srcNum];
                    ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

                    for (int channelNum = 0; channelNum < 3; channelNum++) {
                        tempMean = 0.f;

                        for (int idx = 0; idx < 1024; idx++, ip++) {
                            tempMean += (DTYPE)*ip / 255;
                        }
                        tempMean            /= 1024;
                        m_aMean[channelNum] += tempMean;
                    }
                }
            }

            for (int channelNum = 0; channelNum < 3; channelNum++) {
                m_aMean[channelNum] /= 50000;
                // printf("m_aMean[%d] : %f\n", channelNum, m_aMean[channelNum]);
            }

            float preTempStddev = 0.f;
            float tempStddev    = 0.f;

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10000; j++) {
                    srcNum = i;
                    imgNum = j;

                    imgSrc = m_aaTrainImageSrcs[srcNum];
                    ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

                    for (int channelNum = 0; channelNum < 3; channelNum++) {
                        tempStddev = 0.f;

                        for (int idx = 0; idx < 1024; idx++, ip++) {
                            preTempStddev = ((DTYPE)*ip / 255) - m_aMean[channelNum];
                            tempStddev   += (preTempStddev * preTempStddev);
                        }
                        tempStddev            /= 1024;
                        m_aStddev[channelNum] += tempStddev;
                    }
                }
            }

            for (int channelNum = 0; channelNum < 3; channelNum++) {
                m_aStddev[channelNum] /= 50000;
                m_aStddev[channelNum]  = sqrt(m_aStddev[channelNum]);
                // printf("m_aStddev[%d] : %f\n", channelNum, m_aStddev[channelNum]);
            }
        } else {
            m_aMean   = new float[CAPACITY_OF_IMAGE];
            m_aStddev = new float[CAPACITY_OF_IMAGE];

            for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++) {
                m_aMean[elementNum]   = 0.f;
                m_aStddev[elementNum] = 0.f;
            }

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10000; j++) {
                    srcNum = i;
                    imgNum = j;

                    imgSrc = m_aaTrainImageSrcs[srcNum];
                    ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

                    for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++, ip++) {
                        m_aMean[elementNum] += (DTYPE)*ip / 255;
                    }
                }
            }

            for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++) {
                m_aMean[elementNum] /= 50000;
                // printf("%f, ", m_aMean[elementNum]);
            }

            float preTempStddev = 0.f;

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10000; j++) {
                    srcNum = i;
                    imgNum = j;

                    imgSrc = m_aaTrainImageSrcs[srcNum];
                    ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

                    for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++, ip++) {
                        preTempStddev          = ((DTYPE)*ip / 255) - m_aMean[elementNum];
                        m_aStddev[elementNum] += (preTempStddev * preTempStddev);
                    }
                }
            }

            for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++) {
                m_aStddev[elementNum] /= 50000;
                m_aStddev[elementNum]  = sqrt(m_aStddev[elementNum]);
                // printf("%f, ", m_aStddev[elementNum]);
            }
        }

        return TRUE;
    }

    int CopyMeanAndStddevFromSrc(float *mean, float *stddev) {
        if (m_isNormalizePerChannelWise) {
            m_aMean   = new float[3];
            m_aStddev = new float[3];

            for (int channelNum = 0; channelNum < 3; channelNum++) {
                m_aMean[channelNum]   = mean[channelNum];
                m_aStddev[channelNum] = stddev[channelNum];
            }
        } else {
            m_aMean   = new float[CAPACITY_OF_IMAGE];
            m_aStddev = new float[CAPACITY_OF_IMAGE];

            for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++) {
                m_aMean[elementNum]   = mean[elementNum];
                m_aStddev[elementNum] = stddev[elementNum];
            }
        }

        return TRUE;
    }

    int GetNormalizationMode() {
        return m_isNormalizePerChannelWise;
    }

    float* GetSetOfMean() {
        return m_aMean;
    }

    float* GetSetOfStddev() {
        return m_aStddev;
    }

    int UseRandomHorizontalFlip() {
        if (m_isTrain) {
            m_useRandomHorizontalFlip = TRUE;
        } else {
            std::cout << "test set cannot use data augmentation function : UseRandomHorizontalFlip" << '\n';
            return FALSE;
        }
        return TRUE;
    }

    int FillshuffledListForFlip() {
        int halfOfBatchSize = m_batchSize / 2;

        // std::cout << "halfOfBatchSize : " << halfOfBatchSize << '\n';

        for (int i = 0; i < halfOfBatchSize; i++) {
            m_shuffledListForFlip.push_back(0);
            m_shuffledListForFlip.push_back(1);
        }

        if (m_batchSize % 2 == 1) m_shuffledListForFlip.push_back(0);
        // std::cout << m_shuffledListForFlip.size() << '\n';

        return TRUE;
    }

    int UseRandomCrop(int padding, int lengthOfWidthAndHeight = LEGNTH_OF_WIDTH_AND_HEIGHT) {
        if (m_isTrain) {
            m_useRandomCrop          = TRUE;
            m_padding                = padding;
            m_lengthOfWidthAndHeight = lengthOfWidthAndHeight;
        } else {
            std::cout << "test set cannot use data augmentation function : UseRandomCrop" << '\n';
            return FALSE;
        }
        return TRUE;
    }

    int FillshuffledListForCrop() {
        int limitOfCropPos      = LEGNTH_OF_WIDTH_AND_HEIGHT + (2 * m_padding) - m_lengthOfWidthAndHeight + 1;
        int twoTimesOfBatchSize = m_batchSize * 2;

        // std::cout << "limitOfCropPos : " << limitOfCropPos << '\n';
        // std::cout << "twoTimesOfBatchSize : " << twoTimesOfBatchSize << '\n';

        for (int cntNum = 0; cntNum < twoTimesOfBatchSize; cntNum++) {
            m_shuffledListForCrop.push_back(cntNum % limitOfCropPos);
            // std::cout << m_shuffledListForFlip.back() << '\n';
        }

        // std::cout << "check" << '\n';

        return TRUE;
    }

    int StartProduce() {
        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, m_bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_work = 1;

        pthread_create(&m_thread, NULL, &CIFAR10Reader::ThreadFunc, (void *)this);

        return TRUE;
    }

    int StopProduce() {
        // some signal
        m_work = 0;
        // terminate every element
        sem_post(&m_empty);
        sem_post(&m_full);

        // thread join
        pthread_join(m_thread, NULL);

        if (m_aaQForData) {
            if (m_aaQForData->size() != 0) {
                int numOfTensor = m_aaQForData->size();

                for (int i = 0; i < numOfTensor; i++) {
                    Tensor<DTYPE> **temp = m_aaQForData->front();
                    m_aaQForData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete[] temp;
                    temp = NULL;
                }
            }
        }

        return TRUE;
    }

    static void* ThreadFunc(void *arg) {
        CIFAR10Reader<DTYPE> *reader = (CIFAR10Reader<DTYPE> *)arg;

        reader->DataPreprocess();

        return NULL;
    }

    int DataPreprocess() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        // semaphore is used
        m_recallnum = 0;

        int imgNum = 0;  // random image of above class
        int srcNum = 0;

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        if (m_isTrain) {
            this->ShuffleClassNum();

            do {
                if (((m_recallnum + 1) * m_batchSize) > 50000) {
                    this->ShuffleClassNum();
                    m_recallnum = 0;
                }
                // std::cout << "m_recallnum : " << m_recallnum << '\n';

                if (m_useRandomCrop) {
                    FillshuffledListForCrop();
                    srand(unsigned(time(0)));
                    random_shuffle(m_shuffledListForCrop.begin(), m_shuffledListForCrop.end(), CIFAR10Reader<DTYPE>::random_generator);
                }

                if (m_useRandomHorizontalFlip) {
                    FillshuffledListForFlip();
                    srand(unsigned(time(0)));
                    random_shuffle(m_shuffledListForFlip.begin(), m_shuffledListForFlip.end(), CIFAR10Reader<DTYPE>::random_generator);
                }

                for (int i = 0; i < m_batchSize; i++) {
                    imgNum = m_shuffledList[i + m_recallnum * m_batchSize];
                    srcNum = imgNum / 10000;
                    imgNum = imgNum % 10000;
                    // std::cout << "srcNum : " << srcNum << " imgNum : " << imgNum << '\n';
                    m_aaSetOfImage->push(this->Image2Tensor(srcNum, imgNum));
                    m_aaSetOfLabel->push(this->Label2Tensor(srcNum, imgNum));
                }
                preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
                preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

                this->AddData2Buffer(preprocessedImages, preprocessedLabels);

                sem_post(&m_mutex);
                sem_post(&m_full);

                // int empty_value = 0;
                // int full_value  = 0;
                //
                // sem_getvalue(&m_empty, &empty_value);
                // sem_getvalue(&m_full,  &full_value);
                //
                // printf("full : %d, empty : %d \n", full_value, empty_value);

                m_recallnum++;
            } while (m_work);
        } else {
            do {
                if (((m_recallnum + 1) * m_batchSize) > 10000) {
                    m_recallnum = 0;
                }

                for (int i = 0; i < m_batchSize; i++) {
                    imgNum = i + m_recallnum * m_batchSize;
                    // std::cout << " imgNum : " << imgNum << '\n';

                    m_aaSetOfImage->push(this->Image2Tensor(imgNum));
                    m_aaSetOfLabel->push(this->Label2Tensor(imgNum));
                }
                preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
                preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

                this->AddData2Buffer(preprocessedImages, preprocessedLabels);

                sem_post(&m_mutex);
                sem_post(&m_full);

                // int empty_value = 0;
                // int full_value  = 0;
                //
                // sem_getvalue(&m_empty, &empty_value);
                // sem_getvalue(&m_full,  &full_value);
                //
                // printf("full : %d, empty : %d \n", full_value, empty_value);

                m_recallnum++;
            } while (m_work);
        }

        return TRUE;
    }

    static int random_generator(int upperbound) {
        srand(time(NULL));
        return rand() % upperbound;
    }

    void ShuffleClassNum() {
        random_shuffle(m_shuffledList.begin(), m_shuffledList.end(), CIFAR10Reader<DTYPE>::random_generator);
    }

    Tensor<DTYPE>* Image2Tensor(int srcNum, int imgNum  /*Address of Image*/) {
        int numOfChannel = 3;
        int heightOfImg  = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int widthOfImg   = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int padding      = 0;

        if (m_useRandomCrop) padding = m_padding;
        int heightOfTensor    = heightOfImg + (2 * padding);
        int widthOfTensor     = widthOfImg + (2 * padding);
        int planeSizeOfTensor = heightOfTensor * widthOfTensor;

        unsigned char *imgSrc = m_aaTrainImageSrcs[srcNum];
        unsigned char *ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, numOfChannel * heightOfTensor * widthOfTensor);

        int idx = 0;

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int heightIdx = 0; heightIdx < heightOfImg; heightIdx++) {
                for (int widthIdx = 0; widthIdx < widthOfImg; widthIdx++, ip++) {
                    idx          = channelNum * planeSizeOfTensor + (heightIdx + padding) * widthOfTensor + (widthIdx + padding);
                    (*temp)[idx] = (DTYPE)*ip / 255;
                    // printf("%d = %d * %d + (%d + %d) * %d + (%d + %d) = ", idx, channelNum, planeSizeOfTensor, heightIdx, padding, widthOfTensor, widthIdx, padding);
                    // std::cout << "idx : " << idx << '\n';
                    // int temp_;
                    // std::cin >> temp_;
                }
            }
        }

        // std::cout << temp->GetShape() << '\n';

        // for (int idx = 0; idx < CAPACITY_OF_IMAGE; idx++, ip++) {
        //// std::cout << idx << '\n';
        // (*temp)[idx] = (DTYPE)*ip / 255;
        //// std::cout << (*temp)[idx] << '\n';
        // }

        if (m_useNormalization) {
            temp = Normalization(temp);
        }

        if (m_useRandomCrop) {
            // std::cout << temp->GetShape() << '\n';
            temp = RandomCrop(temp);
        }

        if (m_useRandomHorizontalFlip) {
            int random = m_shuffledListForFlip.back();
            m_shuffledListForFlip.pop_back();
            // std::cout << random << '\n';

            if (random) HorizontalFlip(temp);
        }

        return temp;
    }

    Tensor<DTYPE>* Label2Tensor(int srcNum, int imgNum  /*Address of Label*/) {
        unsigned char *imgSrc = m_aaTrainImageSrcs[srcNum];
        unsigned char *ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG];

        // std::cout << (int)*ip << '\n';

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 10);
        (*temp)[*ip] = (DTYPE)1;

        return temp;
    }

    Tensor<DTYPE>* Image2Tensor(int imgNum  /*Address of Image*/) {
        int channel = 3;
        int height  = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int width   = LEGNTH_OF_WIDTH_AND_HEIGHT;

        unsigned char *ip = &m_aTestImageSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, channel * height * width);

        for (int idx = 0; idx < CAPACITY_OF_IMAGE; idx++, ip++) {
            (*temp)[idx] = (DTYPE)*ip / 255;
        }

        if (m_useNormalization) {
            temp = Normalization(temp);
        }

        return temp;
    }

    Tensor<DTYPE>* Label2Tensor(int imgNum  /*Address of Label*/) {
        unsigned char *ip = &m_aTestImageSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG];

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 10);
        (*temp)[*ip] = (DTYPE)1;

        return temp;
    }

    Tensor<DTYPE>* ConcatenateImage(queue<Tensor<DTYPE> *> *setOfImage) {
        // Do not consider efficiency
        int singleImageSize = setOfImage->front()->GetCapacity();

        Tensor<DTYPE> *result      = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, singleImageSize);
        Tensor<DTYPE> *singleImage = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++) {
            singleImage = setOfImage->front();
            setOfImage->pop();

            for (int idxOfImage = 0; idxOfImage < singleImageSize; idxOfImage++) {
                int idxOfResult = batchNum * singleImageSize + idxOfImage;
                (*result)[idxOfResult] = (*singleImage)[idxOfImage];
            }

            // dealloc single image
            delete singleImage;
            singleImage = NULL;
        }

        // setOfImage->clear();

        return result;
    }

    Tensor<DTYPE>* ConcatenateLabel(queue<Tensor<DTYPE> *> *setOfLabel) {
        // Do not consider efficiency
        Tensor<DTYPE> *result      = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, 10);
        Tensor<DTYPE> *singleLabel = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++) {
            singleLabel = setOfLabel->front();
            setOfLabel->pop();

            for (int idxOfLabel = 0; idxOfLabel < 10; idxOfLabel++) {
                int idxOfResult = batchNum * 10 + idxOfLabel;
                (*result)[idxOfResult] = (*singleLabel)[idxOfLabel];
            }

            // dealloc single image
            delete singleLabel;
            singleLabel = NULL;
        }

        // setOfLabel->clear();

        return result;
    }

    Tensor<DTYPE>* Normalization(Tensor<DTYPE> *image) {
        int numOfChannel = 3;
        int heightOfImg  = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int widthOfImg   = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int padding      = 0;

        if (m_useRandomCrop) padding = m_padding;
        int heightOfTensor    = heightOfImg + (2 * padding);
        int widthOfTensor     = widthOfImg + (2 * padding);
        int planeSizeOfTensor = heightOfTensor * widthOfTensor;

        if (m_isNormalizePerChannelWise) {
            int idx = 0;

            for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
                for (int heightIdx = 0; heightIdx < heightOfImg; heightIdx++) {
                    for (int widthIdx = 0; widthIdx < widthOfImg; widthIdx++) {
                        idx            = channelNum * planeSizeOfTensor + (heightIdx + padding) * widthOfTensor + (widthIdx + padding);
                        (*image)[idx] -= m_aMean[channelNum];
                        (*image)[idx] /= m_aStddev[channelNum];
                    }
                }
            }
        } else {
            int idx                = 0;
            int idxOfMeanAdnStddev = 0;

            for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
                for (int heightIdx = 0; heightIdx < heightOfImg; heightIdx++) {
                    for (int widthIdx = 0; widthIdx < widthOfImg; widthIdx++) {
                        idx                = channelNum * planeSizeOfTensor + (heightIdx + padding) * widthOfTensor + (widthIdx + padding);
                        idxOfMeanAdnStddev = channelNum * planeSizeOfTensor + heightIdx * widthOfImg + widthIdx;
                        (*image)[idx]     -= m_aMean[channelNum];
                        (*image)[idx]     /= m_aStddev[channelNum];
                    }
                }
            }
        }

        return image;
    }

    Tensor<DTYPE>* HorizontalFlip(Tensor<DTYPE> *image) {
        int numOfChannel        = 3;
        int imageSizePerChannel = m_lengthOfWidthAndHeight * m_lengthOfWidthAndHeight;
        int rowSizePerPlane     = m_lengthOfWidthAndHeight;
        int colSizePerPlane     = m_lengthOfWidthAndHeight;
        int halfColSizePerPlane = colSizePerPlane / 2;

        DTYPE temp          = 0;
        int   idx           = 0;
        int   idxOfOpposite = 0;

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int rowNum = 0; rowNum < rowSizePerPlane; rowNum++) {
                for (int colNum = 0; colNum < halfColSizePerPlane; colNum++) {
                    idx           = channelNum * imageSizePerChannel + rowNum * colSizePerPlane + colNum;
                    idxOfOpposite = channelNum * imageSizePerChannel + rowNum * colSizePerPlane + (colSizePerPlane - colNum - 1);
                    // std::cout << "idx : " << idx << " idxOfOpposite : " << idxOfOpposite << '\n';
                    temp                    = (*image)[idx];
                    (*image)[idx]           = (*image)[idxOfOpposite];
                    (*image)[idxOfOpposite] = temp;
                    // std::cin >> temp;
                }
            }
        }

        return image;
    }

    Tensor<DTYPE>* RandomCrop(Tensor<DTYPE> *srcImage) {
        int numOfChannel           = 3;
        int heightOfSrcImg         = LEGNTH_OF_WIDTH_AND_HEIGHT + (2 * m_padding);
        int widthOfSrcImg          = LEGNTH_OF_WIDTH_AND_HEIGHT + (2 * m_padding);
        int srcImageSizePerChannel = heightOfSrcImg * widthOfSrcImg;
        int heightOfImg            = m_lengthOfWidthAndHeight;
        int widthOfImg             = m_lengthOfWidthAndHeight;
        int imageSizePerChannel    = heightOfImg * widthOfImg;

        Tensor<DTYPE> *cropedImg = Tensor<DTYPE>::Zeros(1, 1, 1, 1, numOfChannel * heightOfImg * widthOfImg);

        // std::cout << cropedImg->GetShape() << '\n';

        int startPosW = m_shuffledListForCrop.back();
        m_shuffledListForCrop.pop_back();
        int startPosH = m_shuffledListForCrop.back();
        m_shuffledListForCrop.pop_back();

        // printf("startPosW : %d, startPosH: %d , rsize : %d\n", startPosW, startPosH, m_shuffledListForCrop.size());

        int srcIdx = 0;
        int idx    = 0;

        // std::cout << srcImage->GetShape()  << '\n';

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int rowNum = 0; rowNum < heightOfImg; rowNum++) {
                for (int colNum = 0; colNum < widthOfImg; colNum++) {
                    srcIdx = channelNum * srcImageSizePerChannel + (startPosH + rowNum) * widthOfSrcImg + (startPosW + colNum);
                    idx    = channelNum * imageSizePerChannel + rowNum * widthOfImg + colNum;
                    // std::cout << "srcImage : " << srcImage->GetShape() << "srcidx : " << srcIdx << '\n';
                    // std::cout << "cropedImg : " << cropedImg->GetShape() << "idx : " << idx << '\n';
                    (*cropedImg)[idx] = (*srcImage)[srcIdx];
                }
            }
        }

        // std::cout << "check" << '\n';

        delete srcImage;

        return cropedImg;
    }

    int AddData2Buffer(Tensor<DTYPE> *setOfImage, Tensor<DTYPE> *setOfLabel) {
        Tensor<DTYPE> **result = new Tensor<DTYPE> *[2];

        result[0] = setOfImage;
        result[1] = setOfLabel;

        m_aaQForData->push(result);

        return TRUE;
    }

    Tensor<DTYPE>** GetDataFromBuffer() {
        sem_wait(&m_full);
        sem_wait(&m_mutex);

        Tensor<DTYPE> **result = m_aaQForData->front();
        m_aaQForData->pop();

        sem_post(&m_mutex);
        sem_post(&m_empty);

        // int empty_value = 0;
        // int full_value  = 0;
        //
        // sem_getvalue(&m_empty, &empty_value);
        // sem_getvalue(&m_full,  &full_value);
        //
        // printf("full : %d, empty : %d \n", full_value, empty_value);

        return result;
    }
};
