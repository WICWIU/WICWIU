#ifndef DATALOADER_H_
#define DATALOADER_H_    value

#include <vector>
#include <queue>
#include <thread>
#include <semaphore.h>
#include <cassert>
#include <algorithm>
#include <cstdlib>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Dataset.hpp"

template<typename DTYPE> class DataLoader {
private:
    /* data */
    Dataset<DTYPE> *m_pDataset;
    int m_lenOfDataset;
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

    bool isWorkerEnded_;

    void Alloc();
    void Delete();

    void Init();

public:
    DataLoader(Dataset<DTYPE> *dataset, int batchSize = 1, int useShuffle = FALSE, int numOfWorker = 1, int dropLast = TRUE);
    virtual ~DataLoader();


    void                          StartProcess();
    void                          StopProcess();

    // distribute data idx to each thread
    virtual void                  DistributeIdxOfData2Thread();
    virtual void                  MakeAllOfIndex(std::vector<int> *pAllOfIndex);

    virtual void                  DataPreprocess();

    void                          Push2IdxBuffer(std::vector<int> *setOfIdx);

    std::vector<int>            * GetIdxSetFromIdxBuffer();

    Tensor<DTYPE>               * Concatenate(std::queue<Tensor<DTYPE> *>& setOfData);

    void                          Push2GlobalBuffer(std::vector<Tensor<DTYPE> *> *preprocessedData);

    virtual std::vector<Tensor<DTYPE> *>* GetDataFromGlobalBuffer();

    void SetWorkingSignal(int signal) { m_nowWorking = signal;}

    int GetBatchSize(){return m_batchSize;}
    int GetWorkingSignal(){return m_nowWorking;}
    int GetNumOfEachDatasetMember(){return m_numOfEachDatasetMember;}
    int GetUseShuffle(){return m_useShuffle;}
    int GetNumOfWorker() { return m_numOfWorker; }
    bool GetIsWorkerEnded_() { return isWorkerEnded_; }
    sem_t* GetGlobalFullAddr() { return  &m_globalFull; }
    sem_t* GetGlobalEmptyAddr() { return &m_globalEmpty; }
    sem_t* GetGlobalMutexAddr() { return &m_globalMutex; }

    std::queue<std::vector<Tensor<DTYPE>*>*>*  GetGlobalBufferAddr() { return &m_globalBuffer; }

    Dataset<DTYPE> * GetDataset(){return m_pDataset;}

    // static int random_generator(int upperbound);
};

static int random_generator(int upperbound) {
    srand(time(NULL));
    return (upperbound == 0) ? 0 : rand() % upperbound;
}

template<typename DTYPE> void DataLoader<DTYPE>::Alloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to asign allocate memory dynamically
    // thread allocate
    m_aaWorkerForProcess = new std::thread[m_numOfWorker];
}

template<typename DTYPE> void DataLoader<DTYPE>::Delete() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
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
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to free memory which was allocated at Alloc()
    m_pDataset               = NULL;
    m_lenOfDataset           = 1;
    m_numOfEachDatasetMember = 1;
    m_aaWorkerForProcess     = NULL;
    m_numOfWorker            = 1;
    m_nowWorking             = FALSE;
    m_batchSize              = 1;
    m_dropLast               = FALSE;
    m_useShuffle             = FALSE;
    this->isWorkerEnded_ = false;
}

template<typename DTYPE> DataLoader<DTYPE>::DataLoader(Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast) {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
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

    // elicit information of data;
    m_lenOfDataset = m_pDataset->GetLength();
    assert(m_lenOfDataset > 0);
    m_numOfEachDatasetMember = m_pDataset->GetNumOfDatasetMember();
    assert(m_numOfEachDatasetMember > 0);

#ifdef __DEBUG___
    std::cout << "m_pDataset: " << m_pDataset << '\n';
    std::cout << "m_batchSize: " << m_batchSize << '\n';
    std::cout << "m_useShuffle: " << m_useShuffle << '\n';
    std::cout << "m_numOfWorker: " << m_numOfWorker << '\n';
    std::cout << "m_dropLast: " << m_dropLast << '\n';
    std::cout << "m_lenOfDataset: " << m_lenOfDataset << '\n';
    std::cout << "m_numOfEachDatasetMember: " << m_numOfEachDatasetMember << '\n';
#endif  // ifdef __DEBUG___

    sem_init(&m_distIdxFull,  0, 0);
    sem_init(&m_distIdxEmpty, 0, m_numOfWorker + 1);
    sem_init(&m_distIdxMutex, 0, 1);

    sem_init(&m_globalFull,   0, 0);
    sem_init(&m_globalEmpty,  0, m_numOfWorker * 2);
    sem_init(&m_globalMutex,  0, 1);

    this->Alloc();
    // this->StartProcess();
}

template<typename DTYPE> DataLoader<DTYPE>::~DataLoader() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to free all dynamic allocated elements
    // this->StopProcess();
    this->Delete();
}

template<typename DTYPE> void DataLoader<DTYPE>::StartProcess() {
    // Generate thread for Dist - DistributeIdxOfData2Thread()
    // Generate thread set for Process -
    m_nowWorking = TRUE;

    // this->DistributeIdxOfData2Thread();

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

    this->isWorkerEnded_ = true;

    sem_post(&m_distIdxEmpty);  // for thread terminate
    m_aThreadForDistInfo.join();
    printf("Join dataloader base thread\r\n");
}


template<typename DTYPE> void DataLoader<DTYPE>::MakeAllOfIndex(std::vector<int> *pAllOfIndex)
{
    pAllOfIndex->resize(m_lenOfDataset);
    for (int i = 0; i < m_lenOfDataset; i++)
		(*pAllOfIndex)[i] = i;
}

template<typename DTYPE> void DataLoader<DTYPE>::DistributeIdxOfData2Thread() {
//    std::vector<int> *allOfIdx = new std::vector<int>(m_lenOfDataset);
    std::vector<int> *allOfIdx = new std::vector<int>();

    this->MakeAllOfIndex(allOfIdx);         // virtual function

    std::vector<int> *setOfIdx = NULL;
    // int dropLastSize           = m_lenOfDataset % m_batchSize; // num of dropLast
    // int numOfBatchBlockSize    = m_lenOfDataset / m_batchSize;
    int dropLastSize           = allOfIdx->size() % m_batchSize; // num of dropLast
    int numOfBatchBlockSize    = allOfIdx->size() / m_batchSize;

    int cnt                    = 0;

    if (m_useShuffle)
        std::random_shuffle(allOfIdx->begin(), allOfIdx->end(), random_generator);

    while (m_nowWorking) {
        setOfIdx = new std::vector<int>(m_batchSize);

        for (int i = 0; i < m_batchSize; i++) {
            (*setOfIdx)[i] = (*allOfIdx)[m_batchSize * cnt + i];
            #ifdef __DEBUG__
            std::cout << "idx:" << m_batchSize * cnt + i << " " << (*setOfIdx)[i] << " ";
            #endif  // __DEBUG__
        }
        cnt++;

        this->Push2IdxBuffer(setOfIdx);

        if (cnt == numOfBatchBlockSize) {
            if (!m_dropLast && dropLastSize) {
                std::reverse(allOfIdx->begin(), allOfIdx->end());

                if (m_useShuffle)
                    std::random_shuffle(allOfIdx->begin() + dropLastSize, allOfIdx->end(), random_generator);
            } else {
                if (m_useShuffle)
                std::random_shuffle(allOfIdx->begin(), allOfIdx->end(), random_generator);
            }
            cnt = 0;
        }
    }

    delete allOfIdx;
}

template<typename DTYPE> void DataLoader<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    // arrange everything for worker
    std::queue<Tensor<DTYPE> *> *localBuffer       = new std::queue<Tensor<DTYPE> *>[m_numOfEachDatasetMember];
    std::vector<int> *setOfIdx                     = NULL;
    int idx                                        = 0;
    std::vector<Tensor<DTYPE> *> *data             = NULL;
    Tensor<DTYPE> *pick                            = NULL;
    std::vector<Tensor<DTYPE> *> *preprocessedData = NULL;
    std::cout << "DataLoader worker" << '\n';
    while (m_nowWorking) {
        // get information from IdxBuffer
        setOfIdx = this->GetIdxSetFromIdxBuffer();

        for (int i = 0; i < m_batchSize; i++) {
            idx = (*setOfIdx)[i];
            // printf("%d", idx);
            data = m_pDataset->GetData(idx);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
            }

            delete data;
            data = NULL;
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

//master랑 합치기위해 만든거
template<typename DTYPE> Tensor<DTYPE> *DataLoader<DTYPE>::Concatenate(std::queue<Tensor<DTYPE> *>& setOfData) {
    // concatenate all preprocessed data into one tensor
    Tensor<DTYPE> *temp   = NULL;
    int capacity          = 1;
    int timesize          = 1;
    Tensor<DTYPE> *result = NULL;
    int dataSize = setOfData.size();
    // We need to consider Timesize

    temp     = setOfData.front();
    capacity = temp->GetCapacity();
    timesize = temp->GetTimeSize();
    int colsize = capacity/timesize;

    result   = Tensor<DTYPE>::Zeros(timesize, m_batchSize, 1, 1, colsize);
    Shape *resultShape = result->GetShape();

    for (int i = 0; i < dataSize; i++) {
        temp = setOfData.front();
        setOfData.pop();

        Shape *tempShape = temp->GetShape();

        for (int ti = 0; ti < timesize; ti++){
            for(int co=0; co < colsize; co++){
              //(*result)[i * capacity + j] = (*temp)[j];

              (*result)[Index5D(resultShape, ti, i, 0, 0, co)] = (*temp)[Index5D(tempShape, ti, 0, 0, 0, co)];
            }
        }

        delete temp;
        temp = NULL;
    }



    // concatenate all data;
    // and pop data on queue;

    return result;
}

//내가 사용하던거
/*
template<typename DTYPE> Tensor<DTYPE> *DataLoader<DTYPE>::Concatenate(std::queue<Tensor<DTYPE> *>& setOfData) {
    // concatenate all preprocessed data into one tensor
    Tensor<DTYPE> *temp   = NULL;
    int capacity          = 1;
    int timesize          = 1;
    Tensor<DTYPE> *result = NULL;
    // We need to consider Timesize

    temp     = setOfData.front();
    capacity = temp->GetCapacity();
    timesize = temp->GetTimeSize();
    int colsize = capacity/timesize;          //timesize가 결국은 1 이여서 문제가 없을거 같음....
    //result   = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, capacity);
    result   = Tensor<DTYPE>::Zeros(timesize, m_batchSize, 1, 1, colsize);

    Shape *resultShape = result->GetShape();

    // std::cout << result->GetShape() << '\n';
    // std::cout << setOfData.size() << '\n';

    for (int i = 0; i < m_batchSize; i++) {
        temp = setOfData.front();
        setOfData.pop();

        Shape *tempShape = temp->GetShape();

        for (int ti = 0; ti < timesize; ti++){
            for(int co=0; co < colsize; co++){
              //(*result)[i * capacity + j] = (*temp)[j];

              (*result)[Index5D(resultShape, ti, i, 0, 0, co)] = (*temp)[Index5D(tempShape, ti, 0, 0, 0, co)];
            }
        }

        delete temp;
        temp = NULL;
    }

    // std::cout << result << '\n';


    // concatenate all data;
    // and pop data on queue;

    return result;
}
*/
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
