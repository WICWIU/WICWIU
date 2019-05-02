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
    std::thread *m_aThreadForDistInfo;
    std::thread **m_aaThreadForProcess;  // dynamic allocation
    int m_numOfWorker;
    int m_nowWorking;

    // for distribute data info
    std::queue<std::vector<int> *> m_splitedInfoBuffer;
    sem_t m_DistInfoFull;  // numOfthread + 1;
    sem_t m_DistInfoEmpty;
    sem_t m_DistInfoMutex;

    int m_batchSize;
    int m_dropLast;  // implement yet
    int m_useShuffle;

    // for global buffer
    std::queue<std::vector<Tensor<DTYPE> *> *> m_globalBuffer;
    sem_t m_globalFull;  // user can define
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

    WData<DTYPE>                * DataPreprocess();

    void                          Push2LocalBuffer();

    std::vector<Tensor<DTYPE> *>* Concatenate();

    void                          Push2GlobalBuffer();

    std::vector<Tensor<DTYPE> *>* GetDataFromGlobalBuffer();
};


template<typename DTYPE> void DataLoader<DTYPE>::Alloc() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to asign allocate memory dynamically
    // thread allocate
    m_aaThreadForProcess = new std::thread *[m_numOfWorker];
}

template<typename DTYPE> void DataLoader<DTYPE>::Delete() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to free memory which was allocated at Alloc()

    if (m_aaThreadForProcess) {
        delete[] m_aaThreadForProcess;
        m_aaThreadForProcess = NULL;
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::Init() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
    // need to free memory which was allocated at Alloc()
    m_pDataset = NULL;
    m_aThreadForDistInfo = NULL;
    m_aaThreadForProcess = NULL;
    m_numOfWorker = 1;
    m_nowWorking = FALSE;

    // m_DistInfoFull = NULL;
    // m_DistInfoEmpty = NULL;
    // m_DistInfoMutex = NULL;

    m_batchSize = 1;
    m_dropLast = FALSE;
    m_useShuffle = FALSE;

    // m_globalFull = NULL;
    // m_globalEmpty = NULL;
    // m_globalMutex = NULL;
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

    for (int i = 0; i < m_numOfWorker; i++) {
        m_aaThreadForProcess[i] = new std::thread([&]() {
            this->DataPreprocess();
        });
        printf("Generate worker[%d] for data preprocessing\r\n", i);
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::StopProcess() {
    // Stop Thread
    // not deallocate!
    m_nowWorking = FALSE;

    for (int i = 0; i < m_numOfWorker; i++) {
        m_aaThreadForProcess[i]->join();
        delete m_aaThreadForProcess[i];
        m_aaThreadForProcess[i] = NULL;
        printf("Join worker[%d]\r\n", i);
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::DistributeIdxOfData2Thread() {
    // shuffle, batch, m_dropLast
}

template<typename DTYPE> WData<DTYPE> *DataLoader<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    while (m_nowWorking) {
        printf("do\r");
    }
}

template<typename DTYPE> void DataLoader<DTYPE>::Push2LocalBuffer() {
    // push Local Buffer
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *DataLoader<DTYPE>::Concatenate() {
    // concatenate all preprocessed data into one tensor
}

template<typename DTYPE> void DataLoader<DTYPE>::Push2GlobalBuffer() {
    // Push Tensor pair to Global buffer
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *DataLoader<DTYPE>::GetDataFromGlobalBuffer() {
    // pop Tensor pair from Global Buffer
}

#endif  // ifndef DATALOADER_H_
