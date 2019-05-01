#ifndef DATALOADER_H_
#define DATALOADER_H_    value

#include <vector>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include "Tensor.hpp"

template<typename DTYPE> class WData {
public:
    WData();
    virtual ~WData();
};

template<typename DTYPE> class WRule {
private:
    /* data */

public:
    WRule();
    virtual ~WRule();
};

template<typename DTYPE> class DataLoader {
private:
    /* data */
    pthread_t *m_aThread;
    int m_numOfThread;

    // for default mode
    char *m_dataPath;
    std::vector<char *> imageName;
    std::vector<int> label;

    // for distribute data info
    std::queue<std::vector<int> *> m_splitedInfoBuffer;
    sem_t m_DistInfoFull;  // numOfthread + 1;
    sem_t m_DistInfoEmpty;
    sem_t m_DistInfoMutex;

    int m_batchSize;
    int m_useDataRand;

    // for global buffer
    std::queue<std::vector<Tensor<DTYPE> *> *> m_globalBuffer;
    sem_t m_globalFull;  // user can define
    sem_t m_globalEmpty;
    sem_t m_globalMutex;

    int m_globalBufferSize;

    void Alloc();
    void Delete();

public:
    DataLoader();
    virtual ~DataLoader();

    void StartProcess();
    void StopProcess();

    // information of the data pair
    // ex) string - int pair
    void                          SetDataPairInfo();

    // distribute data inforamtion to each thread
    void                          DistributeDataInfo();

    virtual void                  DataPreprocessForThread();

    virtual WData<DTYPE>        * LoadData(  /*rule*/);

    WData<DTYPE>                * AugmentData();

    void                          Push2LocalBuffer(int idx);

    std::vector<Tensor<DTYPE> *>* Concatenate();

    void                          Push2GlobalBuffer();

    std::vector<Tensor<DTYPE> *>* GetDataFromGlobalBuffer();
};


#endif  // ifndef DATALOADER_H_
