#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "Utils.hpp"

#include "KNearestNeighbor.hpp"

// #define  __DEBUG__

// defined in main.cpp
// int LogMessageF(const char *logFile, int bOverwrite, const char *format, ...);
// int LogFeature(const char *fileName, int bOverwrite, int dim, float *data, int width = 0);
// int LogImage(const char *fileName, int bOverwrite, int width, int height, float *data);

// int DisplayImage(int width, int height, float *data);
// int DisplayFeature(int dim, float *data, int width = 0);

KNearestNeighbor::KNearestNeighbor(int dim, int noClass, int noRef, int* pRefLabel,
                                   float* pRefVector[])
{
    try
    {
        m_vRefLabel.resize(noRef);
        m_vRefVector.resize(noRef);
    }
    catch (...)
    {
        printf("Failed to allocate memory (dim = %d, noClass = %d) in %s (%s %d)\n", dim, noClass,
               __FUNCTION__, __FILE__, __LINE__);
        return;
    }

    for (int i = 0; i < noRef; i++)
    {
        m_vRefLabel[i] = pRefLabel[i];

        m_vRefVector[i] = new float[dim];
        if (m_vRefVector[i] == NULL)
        {
            for (int j = 0; j < i; j++)
                delete[] m_vRefVector[i];

            printf("Failed to allocate memory (dim = %d, noClass = %d) in %s (%s %d)\n", dim,
                   noClass, __FUNCTION__, __FILE__, __LINE__);
            return;
        }

        for (int j = 0; j < dim; j++)
            m_vRefVector[i][j] = pRefVector[i][j];
    }

    m_dim = dim;
    m_noClass = noClass;

#ifdef __DEBUG__
    printf("dim = %d, noClass = %d, noRef = %d\n", dim, noClass, noRef);
    for (int i = 0; i < noRef; i++)
    {
        printf("%d : ", m_vRefLabel[i]);
        for (int j = 0; j < dim; j++)
            printf("%f ", m_vRefVector[i][j]);
        printf("\n");
    }
    // printf("Press Enter to continue...");
    // fflush(stdout);
    // getchar();
#endif //  __DEBUG__
}

KNearestNeighbor::~KNearestNeighbor()
{
    for (int i = 0; i < m_noClass; i++)
        delete[] m_vRefVector[i];
}

void KNearestNeighbor::AddReference(int label, float* pRefVector)
{
    float* newRef = new float[m_dim];
    if (newRef == NULL)
    {
        printf("Failed to allocate memory (dim = %d, noClass = %d) in %s (%s %d)\n", m_dim,
               m_noClass, __FUNCTION__, __FILE__, __LINE__);
        return;
    }

    m_vRefLabel.push_back(label);

    memcpy(newRef, pRefVector, m_dim * sizeof(newRef[0]));
    m_vRefVector.push_back(newRef);

    if (label >= m_noClass)
        m_noClass = label + 1;
}

int KNearestNeighbor::Recognize(float* pInput, int k)
{
    int i = 0;
    float minDist = FLT_MAX;

    std::vector<float> vMinDist2;
    std::vector<int> vMinLabel;

    vMinDist2.reserve(k);
    vMinDist2.resize(0);

    vMinLabel.reserve(k);
    vMinLabel.resize(0);

    int noRef = m_vRefVector.size();

#ifdef __DEBUG__
    printf("Starting %s, m_noClass = %d, noRef = %d...\n", __FUNCTION__, m_noClass, noRef);
    printf("pInput (dim = %d): ", m_dim);
    DisplayFeature(m_dim, pInput);
#endif //  __DEBUG__

    // find top-k classes
    for (i = 0; i < noRef; i++)
    {
#ifdef __DEBUG__
        printf("Comparing with m_vRefVector[%d]: ", i);
        DisplayFeature(m_dim, m_vRefVector[i]);
#endif //  __DEBUG__

        float dist2 = GetSquareDistance(m_dim, pInput, m_vRefVector[i]);

        int p = 0;
        for (p = vMinDist2.size(); p > 0 && dist2 < vMinDist2[p - 1]; p--)
            ;

#ifdef __DEBUG__
        printf("dist2 with %d (label = %d) = %f, p = %d\n", i, m_vRefLabel[i], dist2, p);
#endif //  __DEBUG__

        if (p < k)
        {
            if (p == vMinDist2.size())
            {
                vMinDist2.push_back(dist2);
                vMinLabel.push_back(m_vRefLabel[i]);
            }
            else
            {
                if (vMinDist2.size() < k)
                    vMinDist2.resize(vMinDist2.size() + 1);
                memmove(&vMinDist2[p + 1], &vMinDist2[p],
                        (vMinDist2.size() - p - 1) * sizeof(vMinDist2[0]));
                vMinDist2[p] = dist2;

                if (vMinLabel.size() < k)
                    vMinLabel.resize(vMinLabel.size() + 1);
                memmove(&vMinLabel[p + 1], &vMinLabel[p],
                        (vMinLabel.size() - p - 1) * sizeof(vMinLabel[0]));
                vMinLabel[p] = m_vRefLabel[i];
            }
        }

#ifdef __DEBUG__
        for (int j = 0; j < vMinDist2.size(); j++)
            printf("ranking %d: %d %f\n", j, vMinLabel[j], vMinDist2[j]);
#endif //  __DEBUG__
    }

#ifdef __DEBUG__
    // printf("Press Enter to continue...");
    // getchar();
#endif //  __DEBUG__

    // voting
    std::vector<int> vVote;
    vVote.resize(m_noClass);
    for (i = 0; i < m_noClass; i++)
        vVote[i] = 0;

    for (i = 0; i < vMinLabel.size(); i++)
    {
#ifdef __DEBUG__
        if (vMinLabel[i] >= m_noClass)
        {
            printf("Error! vMinLabel[%d] = %d, m_noClass = %d\n", i, vMinLabel[i], m_noClass);
            printf("Press Enter to continue... (%s)", __FUNCTION__);
            fflush(stdout);
            getchar();
        }
#endif //  __DEBUG__
        vVote[vMinLabel[i]]++;
    }

    // search for the most voted class
    int maxIdx = 0;

    for (i = 1; i < vVote.size(); i++)
    {
        if (vVote[i] > vVote[maxIdx])
            maxIdx = i;
    }

#ifdef __DEBUG__
    printf("maxIdx = %d\n", maxIdx);
#endif //  __DEBUG__

    return maxIdx;
}

float KNearestNeighbor::GetAccuracy(int noSamples, int* pLabels, float* pVectors[], int k)
{
    if (noSamples == 0)
        return 0.F;

    int noCorrect = 0;
    // int noCorrect = 0;

    for (int i = 0; i < noSamples; i++)
    {
        int result = Recognize(pVectors[i], k);
        if (result == pLabels[i])
            noCorrect++;
    }

    return noCorrect / (float)noSamples;
}

#ifdef OLD_CODE
void knn_test();

void knn_test()
{
    int noClass = 5;
    int dim = 5;

    int aRefLabel[10] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};

    float aaRef[10][5] = {{1.F, 0.F, 0.F, 0.F, 0.F},  {0.F, 1.F, 0.F, 0.F, 0.F},
                          {0.F, 0.F, 1.F, 0.F, 0.F},  {0.F, 0.F, 0.F, 1.F, 0.F},
                          {0.F, 0.F, 0.F, 0.F, 1.F},  {1.1F, 0.F, 0.F, 0.F, 0.F},
                          {0.F, 1.1F, 0.F, 0.F, 0.F}, {0.F, 0.F, 1.1F, 0.F, 0.F},
                          {0.F, 0.F, 0.F, 1.1F, 0.F}, {0.F, 0.F, 0.F, 0.F, 1.1F}};

    float* paRef[10];
    for (int i = 0; i < 10; i++)
        paRef[i] = aaRef[i];

    int aTestLabel[3] = {0, 3, 4};

    float aaTest[3][5] = {
        {0.7F, 0.1F, 0.F, 0.F, 0.F}, {0.F, 0.1F, 0.8F, 0.F, 0.F}, {0.F, 0.F, 0.F, 0.F, 1.2F}};

    float* paTest[3];
    for (int i = 0; i < 3; i++)
        paTest[i] = aaTest[i];

    KNearestNeighbor knn(dim, noClass, 10, aRefLabel, paRef);

    for (int i = 0; i < 3; i++)
    {
        int result = knn.Recognize(aaTest[i], 3);
        printf("===== %d: result = %d\n", i, result);
    }

    float accuracy = knn.GetAccuracy(3, aTestLabel, paTest);
    printf("accuracy = %f\n", accuracy);
}

#endif //  OLD_CODE
