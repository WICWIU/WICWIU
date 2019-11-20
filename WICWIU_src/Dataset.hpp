#ifndef _DATASET_H_
#define _DATASET_H_

#include <stdio.h>
#include <vector>
#include <queue>

#include "Tensor.hpp"

template<typename DTYPE> class WData {
public:
    DTYPE *m_aData;
    int m_capacity;

    WData(DTYPE *data, int capacity) {
        m_aData    = data;
        m_capacity = capacity;
    }

    virtual ~WData() {
        delete[] m_aData;
    }

    virtual DTYPE* GetData() {
        return m_aData;
    }

    virtual int GetCapacity() {
        return m_capacity;
    }

    DTYPE& operator[](int idx) {
        return m_aData[idx];
    }
};

template<typename DTYPE> class Dataset {  // [] operator override
private:
    /* data */
    char m_dataPath[256];
    std::vector<char *> imageName;
    std::vector<int> label;
	std::vector<int> m_vPosIndex;
	std::vector<int> m_vNegIndex;

public:
    Dataset();
    virtual ~Dataset();

    virtual void                          Alloc();
    virtual void                          Dealloc();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);
    virtual std::vector<Tensor<DTYPE> *>* GetDataOfPositiveLabel(int anchorIdx, int *pPosIdx = NULL);
    virtual std::vector<Tensor<DTYPE> *>* GetDataOfNegativeLabel(int anchorIdx, int *pNegIdx = NULL);

    void                                  SetLabel(const int *pLabel, int noLabel);
    void                                  SetLabel(const unsigned char *pLabel, int noLabel);
    virtual int 						  GetLabel(int idx) {
		if(idx < 0 || idx >= label.size()){
            printf("idx = %d is out of range (label.size() = %lu) in %s (%s %d)\n", idx, label.size(), __FUNCTION__, __FILE__, __LINE__);
            MyPause(__FUNCTION__);
            return -1;
        }

		return label[idx];
    }
    virtual int                           GetLength() { return label.size(); }
    int                                   GetNumOfDatasetMember();
    virtual void                           CopyData(int idx, DTYPE *pDest) {             // copy i-th iamge into pDest. (designed for k-NN)
        printf("This functions should be overriden by derived class");
        printf("Press Enter to continue... (%s)", __FUNCTION__);
        getchar();
    }   

	virtual void 						 SetPosNegIndices(std::vector<int> *pvPosIndex, std::vector<int> *pvNegIndex){	// registers indices fo positive and negative samples for each sample
		if(pvPosIndex && pvPosIndex->size() > 0){
			m_vPosIndex.resize(pvPosIndex->size());
			memcpy(&m_vPosIndex[0], &(*pvPosIndex)[0], pvPosIndex->size() * sizeof(m_vPosIndex[0]));
		}
		if(pvNegIndex && pvNegIndex->size() > 0){
			m_vNegIndex.resize(pvNegIndex->size());
			memcpy(&m_vNegIndex[0], &(*pvNegIndex)[0], pvNegIndex->size() * sizeof(m_vNegIndex[0]));
		}
	}

    std::vector<int> &                   GetPositiveIndices() { return m_vPosIndex; }
    std::vector<int> &                   GetNegativeIndices() { return m_vNegIndex; }

	virtual int 						 GetPositiveIndex(int idx){			// for triplet loss
        if(rand() % 2 != 0)         // for stochasticity
            return -1;
		return (idx < m_vPosIndex.size() ? m_vPosIndex[idx] : -1);
	}

	virtual int 						 GetNegativeIndex(int idx){			// for triplet loss
        if(rand() % 2 != 0)         // for stochasticity
            return -1;
		return (idx < m_vNegIndex.size() ? m_vNegIndex[idx] : -1);
	}    
};

template<typename DTYPE> Dataset<DTYPE>::Dataset() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> Dataset<DTYPE>::~Dataset() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> void Dataset<DTYPE>::Alloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> void Dataset<DTYPE>::Dealloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *Dataset<DTYPE>::GetData(int idx) {
    // virtual
    // we need to implement default function
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(1, NULL);
    int capacity                         = 1;

    Tensor<DTYPE> *data = Tensor<DTYPE>::Zeros(1, 1, 1, 1, capacity);
    (*data)[0]   = (DTYPE)idx;
    (*result)[0] = data;

    return result;
}

template<typename DTYPE> int Dataset<DTYPE>::GetNumOfDatasetMember() {
// # of data columns
// e.g. (X, Y) : returns 2
// e.g. (X, Y, Z): returns 3

    // virtual
    // we need to implement default function
    int numOfDatasetMember = 0;

    std::vector<Tensor<DTYPE> *> *temp = this->GetData(0);

    if (temp) {
        numOfDatasetMember = temp->size();

        for (int i = 0; i < numOfDatasetMember; i++) {
            if ((*temp)[i]) {
                delete (*temp)[i];
                (*temp)[i] = NULL;
            }
        }
        delete temp;
        temp = NULL;
    }


    return numOfDatasetMember;
}

template<typename DTYPE> void Dataset<DTYPE>::SetLabel(const int *pLabel, int noLabel)
{
    try {
        label.resize(noLabel);
    } catch(...){
        printf("Failed to allocate memory (noLabel = %d) in %s (%s %d)\n", noLabel, __FUNCTION__, __FILE__, __LINE__);
        return;
    }
//    memcpy_s(&label[0], noLabel * sizeof(int), pLabel, noLabel * sizeof(int))

    for(int i = 0; i < noLabel; i++)
        label[i] = pLabel[i];

#ifdef __DEBUG__
    printf("SetLabel() read %d labels of int type\n", noLabel);
//  printf("Press Enter to continue...");
//  getchar();
#endif  // __DEBUG__
}

template<typename DTYPE> void Dataset<DTYPE>::SetLabel(const unsigned char *pLabel, int noLabel)
{
    try {
        label.resize(noLabel);
    } catch(...){
        printf("Failed to allocate memory (noLabel = %d) in %s (%s %d)\n", noLabel, __FUNCTION__, __FILE__, __LINE__);
        return;
    }

    for(int i = 0; i < noLabel; i++)
        label[i] = (int)pLabel[i];

#ifdef __DEBUG__
// for test
// FILE *fp = fopen("label.txt", "w");
// for(int j = 0; j < noLabel; j++){
//     fprintf(fp, "%d\t%d\n", j, label[j]);
// }
// fclose(fp);

    printf("SetLabel() read %d labels of unsigned char type\n", noLabel);
//  printf("Press Enter to continue...");
//  getchar();
#endif  // __DEBUG__
}


template<typename DTYPE> std::vector<Tensor<DTYPE>*> *Dataset<DTYPE>::GetDataOfPositiveLabel(int anchorIdx, int *pPosIdx)
{
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    int anchorLabel = this->GetLabel(anchorIdx);
    if(anchorLabel < 0){
        printf("Error! The label of anchor sample (idx = %d) is %d in %s (%s %d)\n", anchorIdx, anchorLabel, __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }


// printf("anchorIdx = %d, anchorLabel = %d\n", anchorIdx, anchorLabel);
	int posIdx = this->GetPositiveIndex(anchorIdx);

	if(posIdx < 0){
   		int noSamples = GetLength();
   		posIdx = rand() % noSamples;
   		for(int i = 0; i < noSamples; i++){
   	  	 	if(this->GetLabel(posIdx) == anchorLabel && posIdx != anchorIdx)
				break;
	
   	     posIdx++;
   	     if(posIdx >= noSamples)
   	         posIdx = 0;
   		}
	}

//printf("posIdx = %d, posLabel = %d\n", m_curImg, this->GetLabel(m_curImg));
//getchar();

    if(pPosIdx)
        *pPosIdx = posIdx;

    return GetData(posIdx);
}

template<typename DTYPE> std::vector<Tensor<DTYPE>*> *Dataset<DTYPE>::GetDataOfNegativeLabel(int anchorIdx, int *pNegIdx)
{
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    int anchorLabel = this->GetLabel(anchorIdx);
    if(anchorLabel < 0){
        printf("Error! The label of anchor sample (idx = %d) is %d in %s (%s %d)\n", anchorIdx, anchorLabel, __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }

	int negIdx = this->GetNegativeIndex(anchorIdx);

	if(negIdx < 0){
		int noSamples = GetLength();
		negIdx = rand() % noSamples;
		for(int i = 0; i < noSamples; i++){
			if(this->GetLabel(negIdx) != anchorLabel)
				break;
			negIdx++;
			if(negIdx >= noSamples)
				negIdx = 0;
		}
	}

    if(pNegIdx)
        *pNegIdx = negIdx;

    return GetData(negIdx);
}

#endif  // ifndef _DATASET_H_
