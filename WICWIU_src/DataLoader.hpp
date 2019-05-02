#ifndef DATALOADER_H_
#define DATALOADER_H_    value

#include <vector>
#include <queue>
#include <thread>
#include <semaphore.h>
#include <cassert>

#include <stdio.h>
#include <stdlib.h>

#include "Dataset.hpp"

template<typename DTYPE> class DataLoader {
private:
    /* data */
    Dataset<DTYPE> *m_pDataset;
    int m_numOfDataset;
    int m_numOfEachDatasetMember;
    std::thread m_aThreadForDistInfo;
    std::thread *m_aaWorkerForProcess;  // dynamic allocation
    int m_numOfWorker;
    int m_nowWorking;

    // for distribute data info
    std::queue<std::vector<int> *> m_splitedIdxBuffer;
    sem_t m_distIdxFull;  // numOfthread + 1;
    sem_t m_distIdxEmpty;
    sem_t m_distIdxMutex;

    int m_batchSize;
    int m_dropLast;  // implement yet
    int m_useShuffle;

    // for global buffer
    std::queue<std::vector<Tensor<DTYPE> *> *> m_globalBuffer;
    sem_t m_globalFull;  // numOfthread * 2
    sem_t m_globalEmpty;
    sem_t m_globalMutex;

    void Alloc();
    void Delete();

    void Init();

public:
    DataLoader(Dataset<DTYPE> *dataset, int batchSize = 1, int useShuffle = FALSE, int numOfWorker = 1, int dropLast = TRUE);
    virtual ~DataLoader();


    void                          StartProcess();
    void                          StopProcess();

    // distribute data idx to each thread
    void                          DistributeIdxOfData2Thread();

    void                          DataPreprocess();

    void                          Push2IdxBuffer(std::vector<int> *setOfIdx);

    std::vector<int>            * GetIdxSetFromIdxBuffer();

    WData<DTYPE>                * ImagePreProcess(WData<DTYPE> *image);

    Tensor<DTYPE>               * Concatenate(std::queue<WData<DTYPE> *>& setOfData);

    void                          Push2GlobalBuffer(std::vector<Tensor<DTYPE> *> *preprocessedData);

    std::vector<Tensor<DTYPE> *>* GetDataFromGlobalBuffer();
};


template<typename DTYPE> void DataLoader<DTYPE>::Alloc() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to asign allocate memory dynamically
    // thread allocate
    m_aaWorkerForProcess = new std::thread[m_numOfWorker];
}

template<typename DTYPE> void DataLoader<DTYPE>::Delete() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to free memory which was allocated at Alloc()

    if (m_aaWorkerForProcess) {
        delete[] m_aaWorkerForProcess;
        m_aaWorkerForProcess = NULL;
    }

    std::vector<int> *temp1 = NULL;

    while (!m_splitedIdxBuffer.empty()) {
        temp1 = m_splitedIdxBuffer.front();
        m_splitedIdxBuffer.pop();
        delete temp1;
        temp1 = NULL;
    }

    std::vector<Tensor<DTYPE> *> *temp2 = NULL;

    while (!m_globalBuffer.empty()) {
        temp2 = m_globalBuffer.front();
        m_globalBuffer.pop();

        if (temp2) {
            for (int i = 0; i < m_numOfEachDatasetMember; i++) {
                if ((*temp2)[i]) {
                    delete (*temp2)[i];
                    (*temp2)[i] = NULL;
                }
            }
        }

        delete temp2;
        temp2 = NULL;
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::Init() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to free memory which was allocated at Alloc()
    m_pDataset               = NULL;
    m_numOfDataset           = 1;
    m_numOfEachDatasetMember = 1;
    m_aaWorkerForProcess     = NULL;
    m_numOfWorker            = 1;
    m_nowWorking             = FALSE;
    m_batchSize              = 1;
    m_dropLast               = FALSE;
    m_useShuffle             = FALSE;
}

template<typename DTYPE> DataLoader<DTYPE>::DataLoader(Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast) {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    this->Init();

    // need to default value to run the data loader (background)
    m_pDataset = dataset;
    // batch size
    m_batchSize = batchSize;
    assert(m_batchSize > 0);
    // random or not
    m_useShuffle = useShuffle;
    // number of thread
    m_numOfWorker = numOfWorker;
    assert(m_numOfWorker > 0);
    // Drop last
    m_dropLast = dropLast;

#ifdef __DEBUG__
    std::cout << m_pDataset << '\n';
    std::cout << m_batchSize << '\n';
    std::cout << m_useShuffle << '\n';
    std::cout << m_numOfWorker << '\n';
    std::cout << m_dropLast << '\n';
#endif  // ifdef __DEBUG__

    // elicit information of data;
    m_numOfDataset = m_pDataset->GetLength();
    assert(m_numOfDataset > 0);
    m_numOfEachDatasetMember = m_pDataset->GetNumOfDatasetMember();
    assert(m_numOfEachDatasetMember > 0);

    sem_init(&m_distIdxFull,  0, 0);
    sem_init(&m_distIdxEmpty, 0, m_numOfWorker + 1);
    sem_init(&m_distIdxMutex, 0, 1);

    sem_init(&m_globalFull,   0, 0);
    sem_init(&m_globalEmpty,  0, m_numOfWorker * 2);
    sem_init(&m_globalMutex,  0, 1);

    this->Alloc();
    this->StartProcess();
}

template<typename DTYPE> DataLoader<DTYPE>::~DataLoader() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to free all dynamic allocated elements
    this->StopProcess();
    this->Delete();
}

template<typename DTYPE> void DataLoader<DTYPE>::StartProcess() {
    // Generate thread for Dist - DistributeIdxOfData2Thread()
    // Generate thread set for Process -
    m_nowWorking = TRUE;

    m_aThreadForDistInfo = std::thread([&]() {
        this->DistributeIdxOfData2Thread();
    });  // lambda expression
    printf("Generate dataloader base thread\r\n");

    for (int i = 0; i < m_numOfWorker; i++) {
        m_aaWorkerForProcess[i] = std::thread([&]() {
            this->DataPreprocess();
        });  // lambda expression
        printf("Generate worker[%d] for data preprocessing\r\n", i);
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::StopProcess() {
    // Stop Thread
    // not deallocate!
    m_nowWorking = FALSE;

    for (int i = 0; i < m_numOfWorker; i++) {
        sem_post(&m_globalEmpty);  // for thread terminate
    }

    for (int j = 0; j < m_numOfWorker; j++) {
        m_aaWorkerForProcess[j].join();
        printf("Join worker[%d]\r\n", j);
    }

    sem_post(&m_distIdxEmpty);  // for thread terminate
    m_aThreadForDistInfo.join();
    printf("Join dataloader base thread\r\n");
}

template<typename DTYPE> void DataLoader<DTYPE>::DistributeIdxOfData2Thread() {
    std::vector<int> *setOfIdx = NULL;

    while (m_nowWorking) {
        setOfIdx = new std::vector<int>(m_batchSize);

        for (int i = 0; i < m_batchSize; i++) {
            (*setOfIdx)[i] = i;
        }

        this->Push2IdxBuffer(setOfIdx);
    }

    // shuffle, batch, m_dropLast
}

template<typename DTYPE> void DataLoader<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    // arrange everything for worker
    std::queue<WData<DTYPE> *> *localBuffer        = new std::queue<WData<DTYPE> *>[m_numOfEachDatasetMember];
    std::vector<int> *setOfIdx                     = NULL;
    int idx                                        = 0;
    std::vector<WData<DTYPE> *> *dataset           = NULL;
    WData<DTYPE> *data                             = NULL;
    std::vector<Tensor<DTYPE> *> *preprocessedData = NULL;

    while (m_nowWorking) {
        // get information from IdxBuffer
        setOfIdx = this->GetIdxSetFromIdxBuffer();

        for (int i = 0; i < m_batchSize; i++) {
            idx = (*setOfIdx)[i];
            printf("%d", idx);
            dataset = m_pDataset->GetData(idx);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                data = (*dataset)[j];
                //
                if (data->GetType() == WDATA_TYPE::IMAGE) {
                    data = this->ImagePreProcess(data);
                }
                // // push data into local buffer
                localBuffer[j].push(data);
            }

            delete dataset;
            dataset = NULL;
        }

        // delete set of idx vector
        delete setOfIdx;
        setOfIdx = NULL;

        preprocessedData = new std::vector<Tensor<DTYPE> *>(m_numOfEachDatasetMember, NULL);  // do not deallocate in this function!

        for (int k = 0; k < m_numOfEachDatasetMember; k++) {
            // concatenate each localbuffer
            // push preprocessedData vector
            (*preprocessedData)[k] = this->Concatenate(localBuffer[k]);
        }

        // push preprocessedData into Global buffer
        this->Push2GlobalBuffer(preprocessedData);
        preprocessedData = NULL;
    }

    delete[] localBuffer;
}

template<typename DTYPE> WData<DTYPE> *DataLoader<DTYPE>::ImagePreProcess(WData<DTYPE> *image) {
    // use preprocessing rule
    return image;
}

template<typename DTYPE> void DataLoader<DTYPE>::Push2IdxBuffer(std::vector<int> *setOfIdx) {
    sem_wait(&m_distIdxEmpty);
    sem_wait(&m_distIdxMutex);

    m_splitedIdxBuffer.push(setOfIdx);

    sem_post(&m_distIdxMutex);
    sem_post(&m_distIdxFull);
}

template<typename DTYPE> std::vector<int> *DataLoader<DTYPE>::GetIdxSetFromIdxBuffer() {
    sem_wait(&m_distIdxFull);
    sem_wait(&m_distIdxMutex);

    std::vector<int> *setOfIdx = m_splitedIdxBuffer.front();
    m_splitedIdxBuffer.pop();

    sem_post(&m_distIdxMutex);
    sem_post(&m_distIdxEmpty);

    return setOfIdx;
}

template<typename DTYPE> Tensor<DTYPE> *DataLoader<DTYPE>::Concatenate(std::queue<WData<DTYPE> *>& setOfData) {
    // concatenate all preprocessed data into one tensor
    WData<DTYPE> *temp    = NULL;
    int capacity          = 1;
    Tensor<DTYPE> *result = NULL;

    // temp = setOfData.front();
    // capacity       = temp.GetCapacity();
    result = Tensor<float>::Zeros(1, m_batchSize, 1, 1, 1);

    return result;
}

template<typename DTYPE> void DataLoader<DTYPE>::Push2GlobalBuffer(std::vector<Tensor<DTYPE> *> *preprocessedData) {
    sem_wait(&m_globalEmpty);
    sem_wait(&m_globalMutex);

    // Push Tensor pair to Global buffer
    m_globalBuffer.push(preprocessedData);

    sem_post(&m_globalMutex);
    sem_post(&m_globalFull);
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *DataLoader<DTYPE>::GetDataFromGlobalBuffer() {
    sem_wait(&m_globalFull);
    sem_wait(&m_globalMutex);

    // pop Tensor pair from Global Buffer
    std::vector<Tensor<DTYPE> *> *preprocessedData = m_globalBuffer.front();
    m_globalBuffer.pop();

    sem_post(&m_globalMutex);
    sem_post(&m_globalEmpty);

    return preprocessedData;
}

#endif  // ifndef DATALOADER_H_
