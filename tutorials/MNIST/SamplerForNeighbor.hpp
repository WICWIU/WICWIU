#ifndef SMAPLERFORNEIGHBOR_H_
#define SMAPLERFORNEIGHBOR_H_    value

#include "../../WICWIU_src/DataLoader.hpp"

template<typename DTYPE>
class SamplerForNeighbor : public DataLoader<DTYPE>{
private:
    /* data */
    int m_numOfClass;
    int m_numOfClassPerBatch;
    std::vector<int> m_labelSet;

public:
    SamplerForNeighbor(int numOfClass, Dataset<DTYPE> *dataset, int batchSize = 1, int useShuffle = FALSE, int numOfWorker = 1, int dropLast = TRUE);
    virtual ~SamplerForNeighbor();

    virtual void DataPreprocess();
};

template<typename DTYPE> SamplerForNeighbor<DTYPE>::SamplerForNeighbor(int numOfClass, Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast)
    : DataLoader<DTYPE>(dataset, batchSize, useShuffle, numOfWorker, dropLast) {
    m_numOfClass = numOfClass;
    m_numOfClassPerBatch = batchSize / numOfClass;
    for(int i = 0; i < numOfClass; i++){
        for(int j = 0; j < m_numOfClassPerBatch; j++){
            m_labelSet.push_back(i);
        }
    }
}

template<typename DTYPE> SamplerForNeighbor<DTYPE>::~SamplerForNeighbor() {}

template<typename DTYPE> void SamplerForNeighbor<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    // arrange everything for worker
    int m_batchSize              = this->GetBatchSize();
    int m_numOfEachDatasetMember = this->GetNumOfEachDatasetMember();

    Dataset<DTYPE> *m_pDataset = this->GetDataset();

    std::queue<Tensor<DTYPE> *> *localBuffer       = new std::queue<Tensor<DTYPE> *>[m_numOfEachDatasetMember];
    std::vector<int> *setOfIdx                     = NULL;
    int label                                      = 0;
    std::vector<Tensor<DTYPE> *> *data             = NULL;
    std::vector<Tensor<DTYPE> *> *preprocessedData = NULL;

    while (this->GetWorkingSignal()) {
        // get information from IdxBuffer

        // Anchor
        for (int i = 0; i < m_batchSize; i++) {
            label = m_labelSet[i] % m_numOfClass;
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

#endif  // ifndef SamplerForNeighbor_H_
