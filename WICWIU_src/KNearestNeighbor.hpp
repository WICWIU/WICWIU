#ifndef __KNearsestNeighbor__
#define __KNearsestNeighbor__

#include <vector>

class KNearestNeighbor {
    int m_dim;
    int m_noClass;

    std::vector<float*> m_vRefVector;
    std::vector<int> m_vRefLabel;
	std::vector<int> m_vRefIndex;

public:
    KNearestNeighbor(){
        m_dim = 0;
        m_noClass = 0;
    }

    KNearestNeighbor(int dim, int noClass, int noRef, int *pRefLabel, float *pRefVector[]);
    virtual ~KNearestNeighbor();

    void AddReference(int label, float *pRefVector);
    int Recognize(float *pInput, int k = 3);

    float GetAccuracy(int noTestSamples, int *pTestLabels, float *pTestVectors[], int k = 3);
};

#endif  // __KNearsestNeighbor__
