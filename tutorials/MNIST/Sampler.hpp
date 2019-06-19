#ifndef SAMPLER_H_
#define SAMPLER_H_    value

#include "../../WICWIU_src/DataLoader.hpp"

template<typename DTYPE>
class Sampler : public DataLoader<DTYPE>{
private:
    /* data */
    int m_numOfClass;

public:
    Sampler(int numOfClass, Dataset<DTYPE> *dataset, int batchSize = 1, int useShuffle = FALSE, int numOfWorker = 1, int dropLast = TRUE);
    virtual ~Sampler();

    virtual void DataPreprocess();
};

template<typename DTYPE> Sampler<DTYPE>::Sampler(int numOfClass, Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast)
    : DataLoader<DTYPE>(dataset, batchSize, useShuffle, numOfWorker, dropLast) {
    m_numOfClass = numOfClass;
}

template<typename DTYPE> Sampler<DTYPE>::~Sampler() {}

template<typename DTYPE> void Sampler<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    // arrange everything for worker
    int m_batchSize              = this->GetBatchSize();
    int numOfAnchorSample        = m_batchSize / 3;
    int numOfPosSample           = numOfAnchorSample;
    int numOfNegSample           = numOfAnchorSample;
    int m_numOfEachDatasetMember = this->GetNumOfEachDatasetMember();

    Dataset<DTYPE> *m_pDataset = this->GetDataset();

    std::queue<Tensor<DTYPE> *> *localBuffer       = new std::queue<Tensor<DTYPE> *>[m_numOfEachDatasetMember];
    std::vector<int> *setOfIdx                     = NULL;
    int label                                      = 0;
    std::vector<Tensor<DTYPE> *> *data             = NULL;
    std::vector<Tensor<DTYPE> *> *preprocessedData = NULL;

    std::queue<int> setOflable;
    std::cout << "SAMPLER worker" << '\n';
    std::cout << "numOfAnchorSample : " << numOfAnchorSample << '\n';

    while (this->GetWorkingSignal()) {
        // get information from IdxBuffer
        setOfIdx = this->GetIdxSetFromIdxBuffer();

        // Anchor
        for (int i = 0; i < numOfAnchorSample; i++) {
            label = (*setOfIdx)[i] % m_numOfClass;
            setOflable.push(label);
            // printf("%d ", label);
            data = m_pDataset->GetData(label);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
                // if(j == 1) std::cout << (*data)[j] << '\n';
            }
            // std::cin >> label;

            delete data;
            data = NULL;
        }
        // printf("\n");

        // Pos
        for (int i = 0; i < numOfPosSample; i++) {
            label = setOflable.front();  // same as anchor
            setOflable.pop();
            setOflable.push(label);
            // printf("%d ", label);
            data = m_pDataset->GetData(label);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
            }

            delete data;
            data = NULL;
        }
        // printf("\n");

        // neg
        for (int i = numOfAnchorSample, j = 0; j < numOfNegSample; i++) {
            label = (*setOfIdx)[i] % m_numOfClass;  // random

            if (setOflable.front() == label) continue;
            else {
                setOflable.pop();
                j++;
            }
            // printf("%d", idx);
            data = m_pDataset->GetData(label);

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

#endif  // ifndef SAMPLER_H_
