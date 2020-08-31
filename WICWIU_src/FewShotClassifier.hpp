#ifndef __FewShotClassifier__
#define __FewShotClassifier__

#include <string>
#include <vector>

#include "NeuralNetwork.hpp"

#include "KNearestNeighbor.hpp"

class FewShotClassifier
{
    std::vector<std::string> m_aClassName;
    int m_noClass;
    int m_inputDim;
    int m_featureDim;

    NeuralNetwork<float>* m_pNN;
    KNearestNeighbor* m_knn; // k-NN classifier, dimension of knn = output dim of m_pNN

public:
    FewShotClassifier(int inputDim, int featureDim, const std::vector<std::string> vClassName,
                      NeuralNetwork<float>* pNN, int noRef, int* pRefLabel, float* pRefSample[],
                      int batchSize);

    FewShotClassifier(int inputDim, int featureDim, const std::vector<std::string> vClassName,
                      NeuralNetwork<float>* pNN, KNearestNeighbor* kNN);
    // dim: sample dim (== input dim of m_pNN, 784 for MNIST)
    // vClassName.size() == <# of classes>
    // length of pLabel = noRefSample

    std::string Recognize(float* pInputSample, int k = 3); // returns the name of the nearest class

    float GetAccuracy(int noTestSample, float* pTestSample[], int* pLabel); // returns accuracy
    // size of pTestSample = noTestSample * dim;
    // length of pLabel = noTestSample

    void AddReference(const char* className, float* pRefSample);
    // className: name of the new class
    // size of pRefSample: sample dim (== input dim of m_pNN, 784 for MNIST)

    int FindClass(const char* className);
};

#endif // __FewShotClassifier__
