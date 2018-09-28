#ifdef __CUDNN__

#ifndef __CUDNN_BATCH_NORMALIZE__
# define __CUDNN_BATCH_NORMALIZE__    value

# include "../Operator.h"

# include <cmath>

template<typename DTYPE>
class CUDNNBatchNormalize : public Operator<DTYPE>{
public:
    CUDNNBatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, std::string pName) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
# if __DEBUG__
        std::cout << "CUDNNBatchNormalize:: CUDNNBatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
# endif // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise, CUDNN_BN_MIN_EPSILON);
    }

    CUDNNBatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pEpsilon, std::string pName) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
# if __DEBUG__
        std::cout << "CUDNNBatchNormalize:: CUDNNBatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, float, std:: string)" << '\n';
# endif // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise, pEpsilon);
    }

    ~CUDNNBatchNormalize() {
# if __DEBUG__
        std::cout << "CUDNNBatchNormalize:: ~ CUDNNBatchNormalize()" << '\n';
# endif // __DEBUG__

        Delete();
    }

# ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        float *CUDNNCachedMean        = NULL;
        float *CUDNNCachedInvVariance = NULL;

        // this->CopyTensorToFloat(m_pTenInput, m_aCUDNNX);
        // this->CopyTensorToFloat(m_pTenScale, m_aCUDNNBnScale);
        // this->CopyTensorToFloat(m_pTenBias, m_aCUDNNBnBias);

        float *CUDNNX       = m_pTenInput->GetGPUData(pTime);
        float *CUDNNBnScale = m_pTenScale->GetGPUData(0);
        float *CUDNNBnBias  = m_pTenBias->GetGPUData(0);

        float *CUDNNY             = m_pTenResult->GetGPUData(pTime);
        float *CUDNNTotalMean     = NULL;
        float *CUDNNTotalVariance = NULL;

        // checkCudaErrors(cudaMalloc(&CUDNNX, (m_inputBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNBnScale, (m_batchSummaryBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNBnBias, (m_batchSummaryBytes)));

        // checkCudaErrors(cudaMalloc(&CUDNNY, (m_inputBytes)));
        checkCudaErrors(cudaMalloc(&CUDNNTotalMean, (m_batchSummaryBytes)));
        checkCudaErrors(cudaMalloc(&CUDNNTotalVariance, (m_batchSummaryBytes)));

        // checkCudaErrors(cudaMemcpy(CUDNNX, m_aCUDNNX, m_inputBytes, cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMemcpy(CUDNNBnScale, m_aCUDNNBnScale, m_batchSummaryBytes, cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMemcpy(CUDNNBnBias, m_aCUDNNBnBias, m_batchSummaryBytes, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(CUDNNTotalMean, m_aCUDNNTotalMean, m_batchSummaryBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(CUDNNTotalVariance, m_aCUDNNTotalVariance, m_batchSummaryBytes, cudaMemcpyHostToDevice));

        switch (m_mode) {
            case Train:
                checkCudaErrors(cudaMalloc(&CUDNNCachedMean, (m_batchSummaryBytes)));
                checkCudaErrors(cudaMalloc(&CUDNNCachedInvVariance, (m_batchSummaryBytes)));
                checkCUDNN(cudnnBatchNormalizationForwardTrain(
                               m_CUDNNHandle,
                               m_CUDNNMode,
                               &m_CUDNNAlpha,
                               &m_CUDNNBeta,
                               m_CUDNNXDesc,
                               CUDNNX,
                               m_CUDNNYDesc,
                               CUDNNY,
                               m_CUDNNBatchSummaryDesc,
                               CUDNNBnScale,
                               CUDNNBnBias,
                               0.0,
                               // CUDNNTotalMean,
                               // CUDNNTotalVariance,
                               NULL,
                               NULL,
                               m_CUDNNEpsilon,
                               // CUDNNCachedMean,
                               // CUDNNCachedInvVariance)
                               NULL,
                               NULL)
                           );
                break;
            case ACCUMULATING:
                checkCUDNN(cudnnBatchNormalizationForwardTrain(
                               m_CUDNNHandle,
                               m_CUDNNMode,
                               &m_CUDNNAlpha,
                               &m_CUDNNBeta,
                               m_CUDNNXDesc,
                               CUDNNX,
                               m_CUDNNYDesc,
                               CUDNNY,
                               m_CUDNNBatchSummaryDesc,
                               CUDNNBnScale,
                               CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor,
                               CUDNNTotalMean,
                               CUDNNTotalVariance,
                               m_CUDNNEpsilon,
                               NULL,
                               NULL)
                           );
                m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));
                break;
            case INFERENCING:
                checkCUDNN(cudnnBatchNormalizationForwardInference(
                               m_CUDNNHandle,
                               m_CUDNNMode,
                               &m_CUDNNAlpha,
                               &m_CUDNNBeta,
                               m_CUDNNXDesc,
                               CUDNNX,
                               m_CUDNNYDesc,
                               CUDNNY,
                               m_CUDNNBatchSummaryDesc,
                               CUDNNBnScale,
                               CUDNNBnBias,
                               CUDNNTotalMean,
                               CUDNNTotalVariance,
                               m_CUDNNEpsilon)
                           );
                break;
            default:
                break;
        }
        checkCudaErrors(cudaDeviceSynchronize());

        // checkCudaErrors(cudaFree(CUDNNX));
        // checkCudaErrors(cudaFree(CUDNNBnScale));
        // checkCudaErrors(cudaFree(CUDNNBnBias));
        // CUDNNX       = NULL;
        // CUDNNBnScale = NULL;
        // CUDNNBnBias  = NULL;

        // checkCudaErrors(cudaMemcpy(m_aCUDNNY, CUDNNY, m_inputBytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_aCUDNNTotalMean, CUDNNTotalMean, m_batchSummaryBytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_aCUDNNTotalVariance, CUDNNTotalVariance, m_batchSummaryBytes, cudaMemcpyDeviceToHost));

        if (m_mode == Train) {
            checkCudaErrors(cudaMemcpy(m_aCUDNNCachedMean, CUDNNCachedMean, m_batchSummaryBytes, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(m_aCUDNNCachedInvVariance, CUDNNCachedInvVariance, m_batchSummaryBytes, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(CUDNNCachedMean));
            checkCudaErrors(cudaFree(CUDNNCachedInvVariance));
            CUDNNCachedMean        = NULL;
            CUDNNCachedInvVariance = NULL;
        }
        // checkCudaErrors(cudaFree(CUDNNY));
        checkCudaErrors(cudaFree(CUDNNTotalMean));
        checkCudaErrors(cudaFree(CUDNNTotalVariance));

        // CUDNNY                 = NULL;
        CUDNNTotalMean     = NULL;
        CUDNNTotalVariance = NULL;

        // for (int i = 0; i < m_inputCapacity; i++) {
        // (*m_pTenResult)[i] = m_aCUDNNY[i];
        // }
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        // this->CopyTensorToFloat(m_pTenDerResult, m_aCUDNNDy);

        float *CUDNNX                 = m_pTenInput->GetGPUData(pTime);
        float *CUDNNBnScale           = m_pTenScale->GetGPUData(0);
        float *CUDNNDy                = m_pTenDerResult->GetGPUData(pTime);
        float *CUDNNCachedMean        = NULL;
        float *CUDNNCachedInvVariance = NULL;

        float *CUDNNDx          = m_pTenDerInput->GetGPUData(pTime);
        float *CUDNNBnScaleDiff = m_pTenDerScale->GetGPUData(pTime);
        float *CUDNNBnBiasDiff  = m_pTenDerBias->GetGPUData(pTime);

        // checkCudaErrors(cudaMalloc(&CUDNNX, (m_inputBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNBnScale, (m_batchSummaryBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNDy, (m_inputBytes)));
        checkCudaErrors(cudaMalloc(&CUDNNCachedMean, (m_batchSummaryBytes)));
        checkCudaErrors(cudaMalloc(&CUDNNCachedInvVariance, (m_batchSummaryBytes)));

        // checkCudaErrors(cudaMalloc(&CUDNNDx, (m_inputBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNBnScaleDiff, (m_batchSummaryBytes)));
        // checkCudaErrors(cudaMalloc(&CUDNNBnBiasDiff, (m_batchSummaryBytes)));

        // checkCudaErrors(cudaMemcpy(CUDNNX, m_aCUDNNX, m_inputBytes, cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMemcpy(CUDNNBnScale, m_aCUDNNBnScale, m_batchSummaryBytes, cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMemcpy(CUDNNDy, m_aCUDNNDy, m_inputBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(CUDNNCachedMean, m_aCUDNNCachedMean, m_batchSummaryBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(CUDNNCachedInvVariance, m_aCUDNNCachedInvVariance, m_batchSummaryBytes, cudaMemcpyHostToDevice));

        checkCUDNN(cudnnBatchNormalizationBackward(
                       m_CUDNNHandle,
                       m_CUDNNMode,
                       &m_CUDNNAlpha,
                       &m_CUDNNBeta,
                       &m_CUDNNAlpha,
                       &m_CUDNNBeta,
                       m_CUDNNXDesc,
                       CUDNNX,
                       m_CUDNNDyDesc,
                       CUDNNDy,
                       m_CUDNNDxDesc,
                       CUDNNDx,
                       m_CUDNNBatchSummaryDesc,
                       CUDNNBnScale,
                       CUDNNBnScaleDiff,
                       CUDNNBnBiasDiff,
                       m_CUDNNEpsilon,
                       // CUDNNCachedMean,
                       // CUDNNCachedInvVariance)
                       NULL,
                       NULL)
                   );
        checkCudaErrors(cudaDeviceSynchronize());

        // checkCudaErrors(cudaFree(CUDNNX));
        // checkCudaErrors(cudaFree(CUDNNBnScale));
        // checkCudaErrors(cudaFree(CUDNNDy));
        checkCudaErrors(cudaFree(CUDNNCachedMean));
        checkCudaErrors(cudaFree(CUDNNCachedInvVariance));
        // CUDNNX                 = NULL;
        // CUDNNBnScale           = NULL;
        // CUDNNDy                = NULL;
        CUDNNCachedMean        = NULL;
        CUDNNCachedInvVariance = NULL;

        // checkCudaErrors(cudaMemcpy(m_aCUDNNDx, CUDNNDx, m_inputBytes, cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaMemcpy(m_aCUDNNBnScaleDiff, CUDNNBnScaleDiff, m_batchSummaryBytes, cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaMemcpy(m_aCUDNNBnBiasDiff, CUDNNBnBiasDiff, m_batchSummaryBytes, cudaMemcpyDeviceToHost));
        //
        // checkCudaErrors(cudaFree(CUDNNDx));
        // checkCudaErrors(cudaFree(CUDNNBnScaleDiff));
        // checkCudaErrors(cudaFree(CUDNNBnBiasDiff));
        // CUDNNDx          = NULL;
        // CUDNNBnScaleDiff = NULL;
        // CUDNNBnBiasDiff  = NULL;

        // for (int i = 0; i < m_batchSummaryCapacity; i++) {
        // (*m_pTenDerInput)[i] = m_aCUDNNDx[i];
        // (*m_pTenDerScale)[i] = m_aCUDNNBnScaleDiff[i];
        // (*m_pTenDerBias)[i]  = m_aCUDNNBnBiasDiff[i];
        // }
        return TRUE;
    }

# endif // if __CUDNN__


    int SetModeTrain() {
        if (m_mode == ACCUMULATING) {
            ;
        } else if (m_mode == INFERENCING) {
            ;
        } else {
            return TRUE;
        }
        m_mode = Train;
        return TRUE;
    }

    int SetModeAccumulate() {
        if (m_mode == Train) {
            // m_numBatch= 0;
            m_CUDNNExponentialAverageFactor = 1.0;

            for (int i = 0; i < m_batchSummaryCapacity; i++) {
                m_aCUDNNTotalMean[i]     = 0.f;
                m_aCUDNNTotalVariance[i] = 0.f;
            }
        } else if (m_mode == INFERENCING) {
            ;
        } else {
            return TRUE;
        }
        m_mode = ACCUMULATING;
        return TRUE;
    }

    int SetModeInference() {
        if (m_CUDNNExponentialAverageFactor < 1.0) {
            if (m_mode == Train) {
                ;
            } else if (m_mode == ACCUMULATING) {
                ;
            } else {
                return TRUE;
            }
        } else {
            std::cout << "failed to set to inference: nothing accumulated." << std::endl;
            return TRUE;
        }
        m_mode = INFERENCING;
        return TRUE;
    }

private:
    Tensor<DTYPE> *m_pTenInput;
    Tensor<DTYPE> *m_pTenScale;
    Tensor<DTYPE> *m_pTenBias;
    Tensor<DTYPE> *m_pTenResult;

    Tensor<DTYPE> *m_pTenDerInput;
    Tensor<DTYPE> *m_pTenDerScale;
    Tensor<DTYPE> *m_pTenDerBias;
    Tensor<DTYPE> *m_pTenDerResult;

    // int m_isChannelwise;

    // float m_epsilon;

    Mode m_mode;
    // int m_numBatch;

    int m_inputCapacity;
    int m_batchSummaryCapacity;

    size_t m_inputBytes;
    size_t m_batchSummaryBytes;

    cudnnHandle_t m_CUDNNHandle;
    cudnnBatchNormMode_t m_CUDNNMode;
    cudnnTensorDescriptor_t m_CUDNNXDesc;
    cudnnTensorDescriptor_t m_CUDNNYDesc;
    cudnnTensorDescriptor_t m_CUDNNDxDesc;
    cudnnTensorDescriptor_t m_CUDNNDyDesc;
    cudnnTensorDescriptor_t m_CUDNNBatchSummaryDesc;

    float m_CUDNNAlpha;
    float m_CUDNNBeta;
    double m_CUDNNEpsilon;
    double m_CUDNNExponentialAverageFactor;

    float *m_aCUDNNX;
    float *m_aCUDNNBnScale;
    float *m_aCUDNNBnBias;

    float *m_aCUDNNY;
    float *m_aCUDNNTotalMean;
    float *m_aCUDNNTotalVariance;
    float *m_aCUDNNCachedMean;
    float *m_aCUDNNCachedInvVariance;

    float *m_aCUDNNDy;
    float *m_aCUDNNBnScaleDiff;
    float *m_aCUDNNBnBiasDiff;
    float *m_aCUDNNDx;

    void Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, double pEpsilon) {
        std::cout << "BatchNormalize:: Alloc( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, double)" << '\n';

        m_pTenInput = pInput->GetResult();
        m_pTenScale = pScale->GetResult();
        m_pTenBias  = pBias->GetResult();

        m_pTenDerInput = pInput->GetDelta();
        m_pTenDerScale = pScale->GetGradient();
        m_pTenDerBias  = pBias->GetGradient();

        Shape *pInputShape    = m_pTenInput->GetShape();
        int    inputBatchSize = m_pTenInput->GetBatchSize();
        int    numChannel     = m_pTenInput->GetChannelSize();
        int    numInputRow    = m_pTenInput->GetRowSize();
        int    numInputColumn = m_pTenInput->GetColSize();

        m_inputCapacity = m_pTenInput->GetCapacity();

        if (pIsChannelwise) {
            m_CUDNNMode            = CUDNN_BATCHNORM_SPATIAL;
            m_batchSummaryCapacity = numChannel;
        } else {
            m_CUDNNMode            = CUDNN_BATCHNORM_PER_ACTIVATION;
            m_batchSummaryCapacity = m_inputCapacity / inputBatchSize;
        }
        m_inputBytes        = m_inputCapacity * sizeof(float);
        m_batchSummaryBytes = m_batchSummaryCapacity * sizeof(float);

        this->SetResult(new Tensor<DTYPE>(new Shape(pInputShape)));
        this->SetDelta(new Tensor<DTYPE>(new Shape(pInputShape)));
        m_pTenResult    = this->GetResult();
        m_pTenDerResult = this->GetDelta();

        // m_epsilon= pEpsilon;

        m_mode = Train;
        // m_numBatch= 0;
        m_CUDNNHandle = this->GetCudnnHandle();

        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNXDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNYDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNDxDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNDyDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNBatchSummaryDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_CUDNNXDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchSize, numChannel, numInputRow, numInputColumn));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_CUDNNYDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchSize, numChannel, numInputRow, numInputColumn));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_CUDNNDxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchSize, numChannel, numInputRow, numInputColumn));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_CUDNNDyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchSize, numChannel, numInputRow, numInputColumn));
        checkCUDNN(cudnnDeriveBNTensorDescriptor(m_CUDNNBatchSummaryDesc, m_CUDNNXDesc, m_CUDNNMode));

        m_CUDNNAlpha                    = 1.f;
        m_CUDNNBeta                     = 0.f;
        m_CUDNNEpsilon                  = pEpsilon;
        m_CUDNNExponentialAverageFactor = 1.0;

        m_aCUDNNX       = NULL;
        m_aCUDNNBnScale = NULL;
        m_aCUDNNBnBias  = NULL;

        m_aCUDNNY                 = NULL;
        m_aCUDNNTotalMean         = NULL;
        m_aCUDNNTotalVariance     = NULL;
        m_aCUDNNCachedMean        = NULL;
        m_aCUDNNCachedInvVariance = NULL;

        m_aCUDNNDy          = NULL;
        m_aCUDNNBnScaleDiff = NULL;
        m_aCUDNNBnBiasDiff  = NULL;
        m_aCUDNNDx          = NULL;

        m_aCUDNNX       = new float[m_inputCapacity];
        m_aCUDNNBnScale = new float[m_batchSummaryCapacity];
        m_aCUDNNBnBias  = new float[m_batchSummaryCapacity];

        m_aCUDNNY                 = new float[m_inputCapacity];
        m_aCUDNNTotalMean         = new float[m_batchSummaryCapacity];
        m_aCUDNNTotalVariance     = new float[m_batchSummaryCapacity];
        m_aCUDNNCachedMean        = new float[m_batchSummaryCapacity];
        m_aCUDNNCachedInvVariance = new float[m_batchSummaryCapacity];

        m_aCUDNNDy          = new float[m_inputCapacity];
        m_aCUDNNBnScaleDiff = new float[m_batchSummaryCapacity];
        m_aCUDNNBnBiasDiff  = new float[m_batchSummaryCapacity];
        m_aCUDNNDx          = new float[m_inputCapacity];

        if (!(m_aCUDNNX
              && m_aCUDNNBnScale
              && m_aCUDNNBnBias
              && m_aCUDNNY
              && m_aCUDNNTotalMean
              && m_aCUDNNTotalVariance
              && m_aCUDNNCachedMean
              && m_aCUDNNCachedInvVariance
              && m_aCUDNNDy
              && m_aCUDNNBnScaleDiff
              && m_aCUDNNBnBiasDiff
              && m_aCUDNNDx)) {
            printf("Failed to Alloc memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            exit(1);
        }
    }

# ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {}

# endif // if __CUDNN__

    void Delete() {
# ifdef __CUDNN__
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNXDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNYDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNDxDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNDyDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNBatchSummaryDesc));

        cudnnDestroy(m_CUDNNHandle);

        delete[] m_aCUDNNX;
        delete[] m_aCUDNNBnScale;
        delete[] m_aCUDNNBnBias;

        delete[] m_aCUDNNY;
        delete[] m_aCUDNNTotalMean;
        delete[] m_aCUDNNTotalVariance;
        delete[] m_aCUDNNCachedMean;
        delete[] m_aCUDNNCachedInvVariance;

        delete[] m_aCUDNNDy;
        delete[] m_aCUDNNBnScaleDiff;
        delete[] m_aCUDNNBnBiasDiff;
        delete[] m_aCUDNNDx;

        m_aCUDNNX       = NULL;
        m_aCUDNNBnScale = NULL;
        m_aCUDNNBnBias  = NULL;

        m_aCUDNNY                 = NULL;
        m_aCUDNNTotalMean         = NULL;
        m_aCUDNNTotalVariance     = NULL;
        m_aCUDNNCachedMean        = NULL;
        m_aCUDNNCachedInvVariance = NULL;

        m_aCUDNNDy          = NULL;
        m_aCUDNNBnScaleDiff = NULL;
        m_aCUDNNBnBiasDiff  = NULL;
        m_aCUDNNDx          = NULL;
# endif // if __CUDNN__
    }

    void CopyTensorToFloat(Tensor<float> *pTensor, float *pFloat) {
        int capacity = pTensor->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            pFloat[i] = (*pTensor)[i];
        }
    }
};

#endif  // __CUDNN_BATCH_NORMALIZE__

#endif  // _CUDNN__
