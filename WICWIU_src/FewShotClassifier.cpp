#include "NeuralNetwork.hpp"

#include "FewShotClassifier.hpp"
#include "KNearestNeighbor.hpp"

FewShotClassifier::FewShotClassifier(int inputDim, int featureDim,
                                     const std::vector<std::string> vClassName,
                                     NeuralNetwork<float>* pNN, int noRef, int* pRefLabel,
                                     float* pRefSample[], int batchSize)
{
    for (int i = 0; i < vClassName.size(); i++)
        m_aClassName.push_back(vClassName[i]);

    m_inputDim = inputDim;
    m_featureDim = featureDim;
    m_noClass = vClassName.size();
    m_pNN = pNN;

    std::vector<float*> vRefFeature;
    AllocFeatureVector(featureDim, noRef, vRefFeature);
    pNN->InputToFeature(inputDim, noRef, pRefSample, featureDim, &vRefFeature[0], batchSize);
    m_knn = new KNearestNeighbor(featureDim, m_noClass, noRef, pRefLabel, &vRefFeature[0]);
    DeleteFeatureVector(vRefFeature);
}

FewShotClassifier::FewShotClassifier(int inputDim, int featureDim,
                                     const std::vector<std::string> vClassName,
                                     NeuralNetwork<float>* pNN, KNearestNeighbor* kNN)
{
    for (int i = 0; i < vClassName.size(); i++)
        m_aClassName.push_back(vClassName[i]);

    m_inputDim = m_featureDim;
    m_noClass = vClassName.size();
    m_pNN = pNN;
    m_knn = kNN;
}

// dim: sample dim (== input dim of m_pNN, 784 for MNIST)
// vClassName.size() == <# of classes>
// length of pLabel = noRefSample
// size of pRefSample = noRefSample * dim

std::string FewShotClassifier::Recognize(float* pInputSample, int k)
{
    std::vector<float*> vInputSample;
    vInputSample.push_back(pInputSample);
    std::vector<float*> vRefFeature;
    AllocFeatureVector(m_featureDim, 1, vRefFeature);
    m_pNN->InputToFeature(m_inputDim, 1, &vInputSample[0], m_featureDim, &vRefFeature[0], 1);
    //  InputToFeature(int inDim, int noSample, float *pSamples[], int outDim, float *pFeatures[],
    //  int batchSize)
    int idx = m_knn->Recognize(vRefFeature[0], k);
    DeleteFeatureVector(vRefFeature);

    return m_aClassName[idx];
} // returns the name of the nearest class

float FewShotClassifier::GetAccuracy(int noTestSample, float* pTestSample[], int* pLabel)
{
    // to do
    return m_knn->GetAccuracy(noTestSample, pLabel, pTestSample, 3);
} // returns accuracy
// size of pTestSample = noTestSample * dim;
// length of pLabel = noTestSample

void FewShotClassifier::AddReference(const char* className, float* pRefSample)
{
    int classIdx = FindClass(className);
    if (classIdx < 0)
    {
        classIdx = m_noClass++;
        m_aClassName.push_back(className);
    }

    std::vector<float*> vRefSample;
    vRefSample.push_back(pRefSample);

    std::vector<float*> vRefFeature;
    AllocFeatureVector(m_featureDim, 1, vRefFeature);
    m_pNN->InputToFeature(m_inputDim, 1, &vRefSample[0], m_featureDim, &vRefFeature[0], 1);

    m_knn->AddReference(classIdx, vRefFeature[0]);
    DeleteFeatureVector(vRefFeature);
}
// className: name of the new class
// size of pRefSample: sample dim (== input dim of m_pNN, 784 for MNIST)

int FewShotClassifier::FindClass(const char* className)
{
    int idx = -1;
    for (int i = 0; i < m_aClassName.size(); i++)
    {
        if (strcmp(className, m_aClassName[i].c_str()) == 0)
            return idx;
    }

    return idx;
}
