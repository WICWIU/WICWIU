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

#include "../../WICWIU_src/Tensor_utils.h"

#define NUMBER_OF_CLASS      1000

#define CAPACITY_OF_IMAGE    3072

#define BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG 3073

#define SIZE_OF_SINGLE_DATA_FILE 30730000

using namespace std;

template<typename DTYPE> class CIFAR10Reader {
private:
    string m_path     = "cifar-10-batches-bin";
    string train_data = "data_batch_";
    string test_data  = "test_batch.bin";

    /*Training image*/
    // for shuffle class index
    vector<int> m_shuffledList;
    // number of img of each class
    unsigned char **m_aaTrainImageSrcs;

    /*Testing image*/
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
    // Random_crop
    int m_useRandomCrop;
    int m_padding;
    int m_lengthOfWidthAndHeight;

    // for normalization
    int m_useNormalization;
    int m_isNormalizePerChannelWise;
    float *m_aMean;
    float *m_aStddev;

    // Flip
    int m_useRandomHorizontalFlip;
    int m_useVerticalHorizontalFlip;

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
        m_lengthOfWidthAndHeight = 32;

        m_useNormalization          = FALSE;
        m_isNormalizePerChannelWise = FALSE;
        m_aMean                     = NULL;
        m_aStddev                   = NULL;

        m_useRandomHorizontalFlip   = FALSE;
        m_useVerticalHorizontalFlip = FALSE;

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
            CalculateTrainingDataMeanAndStddev();
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

    int CalculateTrainingDataMeanAndStddev() {
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
        return rand() % upperbound;
    }

    void ShuffleClassNum() {
        srand(unsigned(time(0)));
        random_shuffle(m_shuffledList.begin(), m_shuffledList.end(), CIFAR10Reader<DTYPE>::random_generator);
    }

    Tensor<DTYPE>* Image2Tensor(int srcNum, int imgNum  /*Address of Image*/) {
        int channel = 3;
        int height  = 32;
        int width   = 32;

        unsigned char *imgSrc = m_aaTrainImageSrcs[srcNum];
        unsigned char *ip     = &imgSrc[imgNum * BASE_SIZE_OF_BYTE_FOR_TO_DIVIDE_IMG + 1];

        // std::cout << (int)*ip << '\n';

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, channel * height * width);

        for (int idx = 0; idx < CAPACITY_OF_IMAGE; idx++, ip++) {
            // std::cout << idx << '\n';
            (*temp)[idx] = (DTYPE)*ip / 255;
            // std::cout << (*temp)[idx] << '\n';
        }

        if (m_useNormalization) {
            temp = Normalization(temp);
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
        int height  = 32;
        int width   = 32;

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

    Tensor<DTYPE>* Normalization(Tensor<DTYPE> *images) {
        int singleImageSize = CAPACITY_OF_IMAGE;

        if (m_isNormalizePerChannelWise) {
            int numOfChannel        = 3;
            int imageSizePerChannel = singleImageSize / numOfChannel;
            int idxOfResult         = 0;

            for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
                for (int idxOfImage = 0; idxOfImage < imageSizePerChannel; idxOfImage++) {
                    idxOfResult             = channelNum * imageSizePerChannel + idxOfImage;
                    (*images)[idxOfResult] -= m_aMean[channelNum];
                    (*images)[idxOfResult] /= m_aStddev[channelNum];
                }
            }
        } else {
            for (int idxOfImage = 0; idxOfImage < singleImageSize; idxOfImage++) {
                (*images)[idxOfImage] -= m_aMean[idxOfImage];
                (*images)[idxOfImage] /= m_aStddev[idxOfImage];
            }
        }

        return images;
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
