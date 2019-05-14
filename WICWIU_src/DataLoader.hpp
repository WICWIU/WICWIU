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

/*!
 * @class DataLoader 데이터를 로드하는 클래스
 * @details [상세설명1 : 마지막에 작성예정]
 * @details [상세설명2 : 마지막에 작성예정]
 */
template<typename DTYPE> class DataLoader {
private:
    /* data */
    Dataset<DTYPE> *m_pDataset;
    ///< 데이터를 담고 있는 리스트 (실제 데이터를 인덱스로 접근하기 위함)
    int m_lenOfDataset;
    ///< 데이터의 갯수
    int m_numOfEachDatasetMember;
    ///< 데이터 멤버(인풋, 라벨)의 갯수
    std::thread m_aThreadForDistInfo;
    ///< 데이터를 분배해주는 thread
    std::thread *m_aaWorkerForProcess;  // dynamic allocation
    ///< 실제로 데이터를 로드해서 buffer로 전달하는 thread들 리스트
    int m_numOfWorker;
    ///< thread의 갯수
    int m_nowWorking;
    ///< thread가 일을 해야하는지를 나타내는 flag

    // for distribute data info
    std::queue<std::vector<int> *> m_splitedIdxBuffer;
    ///< 어떤 인덱스를 가져갈지를 담고 있는 버퍼
    sem_t m_distIdxFull;  // numOfthread + 1;
    sem_t m_distIdxEmpty;
    sem_t m_distIdxMutex;

    int m_batchSize;
    int m_dropLast;  // implement yet
    ///< batch단위로 접근하고 남은 데이터를 버릴지 사용할지를 나타내는 변수
    int m_useShuffle;
    ///< shuffle할 것인지를 나타내는 변수(train_dataset의 경우 true)

    // for global buffer
    std::queue<std::vector<Tensor<DTYPE> *> *> m_globalBuffer;
    ///< 각 thread의 로드한 데이터를 모아둔 버퍼 (main에서 이 버퍼에 접근해 사용함)
    sem_t m_globalFull;  // numOfthread * 2
    sem_t m_globalEmpty;
    sem_t m_globalMutex;

    void Alloc();
    void Delete();

    void Init();
    void ElicitInfo();

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

    Tensor<DTYPE>               * Concatenate(std::queue<Tensor<DTYPE> *>& setOfData);

    void                          Push2GlobalBuffer(std::vector<Tensor<DTYPE> *> *preprocessedData);

    std::vector<Tensor<DTYPE> *>* GetDataFromGlobalBuffer();

    // static int random_generator(int upperbound);
};


/*!
 * @brief 랜덤한 숫자를 생성하는 메서드
 * @details 파라미터로 받은 값의 범위 내에서 랜덤한 숫자를 반환해준다.
 * @param upperbound rand_max값
 * @return (static한)랜덤한 숫자
 */
static int random_generator(int upperbound) {
    srand(time(NULL));
    return (upperbound == 0) ? 0 : rand() % upperbound;
}

/*!
 * @brief DataLoader 클래스를 동적 할당하는 메소드
 * @details 사용할 thread갯수만큼을 메모리에 동적으로 할당한다.
 * @return 없음
 */
template<typename DTYPE> void DataLoader<DTYPE>::Alloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to asign allocate memory dynamically
    // thread allocate
    m_aaWorkerForProcess = new std::thread[m_numOfWorker];
}

/*!
 * @brief thread들이 사용한 공간을 반환하는 메소드
 * @details DataLoader클래스 생성때 할당받았던 thread공간을 반납한다.
 * @return 없음
 */
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

/*!
 * @brief DataLoader클래스의 각 멤버 변수를 초기화하는 메소드
 * @details DataLoader클래스의 각 멤버 변수가 가지고 있던 값들을 초기화한다.
 * @return 없음
 */
template<typename DTYPE> void DataLoader<DTYPE>::Init() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to free memory which was allocated at Alloc()
    m_pDataset               = NULL;  //
    m_lenOfDataset           = 1;
    m_numOfEachDatasetMember = 1;
    m_aaWorkerForProcess     = NULL;  //
    m_numOfWorker            = 1;
    m_nowWorking             = FALSE; //
    m_batchSize              = 1;
    m_dropLast               = FALSE; //
    m_useShuffle             = FALSE; //
}

/*!
 * @brief
 * @details
 * @return
 */
template<typename DTYPE> void DataLoader<DTYPE>::ElicitInfo() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to free memory which was allocated at Alloc()
    // elicit information of data;
    m_lenOfDataset = m_pDataset->GetLength();
    assert(m_lenOfDataset > 0);
    m_numOfEachDatasetMember = m_pDataset->GetNumOfDatasetMember();
    assert(m_numOfEachDatasetMember > 0);
}

/*!
 * @brief DataLoader의 생성자
 * @details 데이터, batchSize, useshuffle, thread갯수, dropLast를 파라미터로 받아
 * @details DataLoader클래스를 생성한다.
 * @param *dataset 데이터셋
 * @param batchSize 데이터를 몇개씩 사용할지를 담은 변수
 * @param useShuffle shuffle할지를 결정하는 변수
 * @param numOfWorker thread갯수
 * @param dropLast batch단위보다 작은 데이터 처리방법
 * @return 없음
 */
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

    this->ElicitInfo();

#ifdef __DEBUG___
    std::cout << m_pDataset << '\n';
    std::cout << m_batchSize << '\n';
    std::cout << m_useShuffle << '\n';
    std::cout << m_numOfWorker << '\n';
    std::cout << m_dropLast << '\n';
    std::cout << m_lenOfDataset << '\n';
    std::cout << m_numOfEachDatasetMember << '\n';
#endif  // ifdef __DEBUG___

    sem_init(&m_distIdxFull,  0, 0);
    sem_init(&m_distIdxEmpty, 0, m_numOfWorker + 1);
    sem_init(&m_distIdxMutex, 0, 1);

    sem_init(&m_globalFull,   0, 0);
    sem_init(&m_globalEmpty,  0, m_numOfWorker * 2);
    sem_init(&m_globalMutex,  0, 1);

    this->Alloc();
    this->StartProcess();
}

/*!
 * @brief DataLoader 클래스 소멸자
 * @details StopProcess 메소드를 호출해서 thread처리를 멈추고 클래스를 소멸시킨다.
 * @return 없음
 */
template<typename DTYPE> DataLoader<DTYPE>::~DataLoader() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
    // need to free all dynamic allocated elements
    this->StopProcess();
    this->Delete();
}

/*!
 * @brief thread를 만들고 DataPreprocess메소드를 불러서 thread를 돌리는 메소드
 * @details m_nowWorking을 TRUE로 바꿔주고 thread를 만든다.
 * @details 그리고 실제 thread를 돌리는 메소드인 DataPreprocess를 호출한다.
 * @return 없음
 */
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

/*!
 * @brief [어떤 메소드인지 타이틀]
 * @details m_nowWorking을 FALSE로 바꿔주고
 * @return 없음
 */
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

/*!
 * @brief [어떤 메소드인지 타이틀]
 * @details [메소드 상세설명]
 * @param [파라미터 변수명1] [상세설명]
 * @param [파라미터 변수명2] [상세설명]
 * @return 없음
 */
template<typename DTYPE> void DataLoader<DTYPE>::DistributeIdxOfData2Thread() {
    std::vector<int> *allOfIdx = new std::vector<int>(m_lenOfDataset);
    std::vector<int> *setOfIdx = NULL;
    int dropLastSize           = m_lenOfDataset % m_batchSize; // num of dropLast
    int numOfBatchBlockSize    = m_lenOfDataset / m_batchSize;
    int cnt                    = 0;

    for (int i = 0; i < m_lenOfDataset; i++) (*allOfIdx)[i] = i;

    if (m_useShuffle) std::random_shuffle(allOfIdx->begin(), allOfIdx->end(), random_generator);

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

        if (numOfBatchBlockSize == cnt) {
            if (!m_dropLast && dropLastSize) {
                std::reverse(allOfIdx->begin(), allOfIdx->end());

                if (m_useShuffle) std::random_shuffle(allOfIdx->begin() + dropLastSize, allOfIdx->end(), random_generator);
            } else {
                if (m_useShuffle) std::random_shuffle(allOfIdx->begin(), allOfIdx->end(), random_generator);
            }
            cnt = 0;
        }
    }
}

/*!
 * @brief thread를 돌리는 메소드
 * @details 임시 localBuffer를 만들고 이 곳에 인덱스 버퍼로 부터 데이터를 받는다.
 * @details 그 다음 preprocessedData에 localBuffer의 데이터 셋을 합쳐서 글로벌 버퍼로 넘긴다.
 * @return 없음
 */
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

template<typename DTYPE> Tensor<DTYPE> *DataLoader<DTYPE>::Concatenate(std::queue<Tensor<DTYPE> *>& setOfData) {
    // concatenate all preprocessed data into one tensor
    Tensor<DTYPE> *temp   = NULL;
    int capacity          = 1;
    Tensor<DTYPE> *result = NULL;
    // We need to consider Timesize

    temp     = setOfData.front();
    capacity = temp->GetCapacity();
    // Shape를 이용해서 Shape대로 만들어질 수 있도록 수정(Shape constructor 잘돌아가는지 확인)
    result = Tensor<DTYPE>::Zeros(1, m_batchSize, temp->GetShape()->GetDim(2), temp->GetShape()->GetDim(3), temp->GetShape()->GetDim(4));

    // std::cout << result->GetShape() << '\n';
    // std::cout << setOfData.size() << '\n';

    for (int i = 0; i < m_batchSize; i++) {
        temp = setOfData.front();
        setOfData.pop();

        for (int j = 0; j < capacity; j++) (*result)[i * capacity + j] = (*temp)[j];

        delete temp;
        temp = NULL;
    }

    // std::cout << result->GetShape() << '\n';
    // std::cout << result << '\n';


    // concatenate all data;
    // and pop data on queue;

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

/*!
 * @brief 글로벌 버퍼에 있는 데이터를 가져오는 메소드
 * @details
 * @return [상세설명]
 */
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
