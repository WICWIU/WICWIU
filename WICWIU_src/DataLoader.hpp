#ifndef DATALOADER_H_
#define DATALOADER_H_    value

#include <vector>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>

#include "Dataset.hpp"

template<typename DTYPE> class DataLoader {
private:
    /* data */
    pthread_t m_aThreadForDistInfo;
    pthread_t *m_aThreadForProcess;  // dynamic allocation
    int m_numOfWorker;

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

    int m_globalBufferSize;

    void Alloc();
    void Delete();

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


template<typename DTYPE> DataLoader<DTYPE>::DataLoader(Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast) {
#ifdef __DEBUG__
    std::cout << "construct DataLoader" << '\n';
#endif  // ifdef __DEBUG__
    // need to default value to run the data loader (background)
    // batch size
    m_batchSize = batchSize;
    // random or not
    m_useShuffle = useShuffle;
    // number of thread
    m_numOfWorker = numOfWorker;
    // Drop last
    m_dropLast = dropLast;

    this->Alloc();
}

template<typename DTYPE> DataLoader<DTYPE>::~DataLoader() {
#ifdef __DEBUG__
    std::cout << "deconstruct DataLoader" << '\n';
#endif  // ifdef __DEBUG__
    // need to free all dynamic allocated elements
    this->Delete();
}

template<typename DTYPE> void DataLoader<DTYPE>::Alloc() {
#ifdef __DEBUG__
    std::cout << "allocate something" << '\n';
#endif  // ifdef __DEBUG__
    // need to asign allocate memory dynamically
    // thread allocate
}

template<typename DTYPE> void DataLoader<DTYPE>::Delete() {
#ifdef __DEBUG__
    std::cout << "deallocate something" << '\n';
#endif  // ifdef __DEBUG__
    // need to free memory which was allocated at Alloc()
}

template<typename DTYPE> void DataLoader<DTYPE>::StartProcess() {
    // Generate thread for Dist - DistributeIdxOfData2Thread()
    // Generate thread set for Process -
}

template<typename DTYPE> void DataLoader<DTYPE>::StopProcess() {
    // Stop Thread
    // not deallocate!
}

template<typename DTYPE> void DataLoader<DTYPE>::DistributeIdxOfData2Thread() {
    // shuffle, batch, m_dropLast
}

template<typename DTYPE> WData<DTYPE> *DataLoader<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
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
