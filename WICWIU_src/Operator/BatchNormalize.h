#ifndef __CUDNN_BATCH_NORMALIZE__
#define __CUDNN_BATCH_NORMALIZE__    value

#include "../Operator.h"

#include <cmath>

template<typename DTYPE>
class BatchNormalize : public Operator<DTYPE>{
private:
    Tensor<DTYPE> *m_pTenInput;
    Tensor<DTYPE> *m_pTenScale;
    Tensor<DTYPE> *m_pTenBias;
    Tensor<DTYPE> *m_pTenResult;

    Tensor<DTYPE> *m_pTenDerInput;
    Tensor<DTYPE> *m_pTenDerScale;
    Tensor<DTYPE> *m_pTenDerBias;
    Tensor<DTYPE> *m_pTenDerResult;

    Tensor<DTYPE> *m_aTenTotalMean;
    Tensor<DTYPE> *m_aTenTotalVariance;

    Tensor<DTYPE> *m_aTenCachedMean;
    Tensor<DTYPE> *m_aTenCachedInvVariance;

    int m_inputTimeSize;
    int m_inputBatchSize;
    int m_numChannel;
    int m_numInputRow;
    int m_numInputColumn;

    int m_isChannelwise;
    Mode m_mode;

    int m_inputCapacity;
    int m_batchSummaryCapacity;

    float m_epsilon;
#ifdef __CUDNN__
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
#endif  // _CUDNN__

public:
    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, std::string pName) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise);
    }

    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pEpsilon, std::string pName) : Operator<DTYPE>(pInput, pScale, pBias, pName) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, float, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise, pEpsilon);
    }

    ~BatchNormalize() {
#if __DEBUG__
        std::cout << "BatchNormalize:: ~ BatchNormalize()" << '\n';
#endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, double pEpsilon = 0.01) {
        std::cout << "BatchNormalize:: Alloc( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, double)" << '\n';

        m_pTenInput = pInput->GetResult();
        m_pTenScale = pScale->GetResult();
        m_pTenBias  = pBias->GetResult();

        m_pTenDerInput = pInput->GetGradient();
        m_pTenDerScale = pScale->GetGradient();
        m_pTenDerBias  = pBias->GetGradient();

        m_inputCapacity = m_pTenInput->GetCapacity();

        Shape *pInputShape = m_pTenInput->GetShape();

        m_inputTimeSize  = m_pTenInput->GetTimeSize();
        m_inputBatchSize = m_pTenInput->GetBatchSize();
        m_numChannel     = m_pTenInput->GetChannelSize();
        m_numInputRow    = m_pTenInput->GetRowSize();
        m_numInputColumn = m_pTenInput->GetColSize();

        m_isChannelwise = pIsChannelwise;

        if (m_isChannelwise) {
            m_batchSummaryCapacity  = m_numChannel;
            m_aTenTotalMean         = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenTotalVariance     = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
        } else {
            m_batchSummaryCapacity  = m_numChannel * m_numInputRow * m_numInputColumn;
            m_aTenTotalMean         = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenTotalVariance     = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, m_numInputRow, m_numInputColumn);
        }

        this->SetResult(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));
        this->SetGradient(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));

        m_pTenResult    = this->GetResult();
        m_pTenDerResult = this->GetGradient();

        m_mode = TRAINING;

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_isChannelwise) {
            m_CUDNNMode = CUDNN_BATCHNORM_SPATIAL;
        } else {
            m_CUDNNMode = CUDNN_BATCHNORM_PER_ACTIVATION;
        }

        m_CUDNNHandle = this->GetCudnnHandle();
        m_CUDNNXDesc  = m_pTenInput->GetDescriptor();
        m_CUDNNYDesc  = m_pTenResult->GetDescriptor();
        m_CUDNNDxDesc = m_pTenDerInput->GetDescriptor();
        m_CUDNNDyDesc = m_pTenDerResult->GetDescriptor();
        checkCUDNN(cudnnCreateTensorDescriptor(&m_CUDNNBatchSummaryDesc));
        checkCUDNN(cudnnDeriveBNTensorDescriptor(m_CUDNNBatchSummaryDesc, m_CUDNNXDesc, m_CUDNNMode));

        m_aTenTotalMean->SetDeviceGPU(idOfDevice);
        m_aTenTotalVariance->SetDeviceGPU(idOfDevice);
        m_aTenCachedMean->SetDeviceGPU(idOfDevice);
        m_aTenCachedInvVariance->SetDeviceGPU(idOfDevice);

        m_CUDNNAlpha                    = 1.f;
        m_CUDNNBeta                     = 0.f;
        m_CUDNNEpsilon                  = CUDNN_BN_MIN_EPSILON;
        m_CUDNNExponentialAverageFactor = 1.0;
    }

#endif  // if __CUDNN__

    void Delete() {
#ifdef __CUDNN__
        checkCUDNN(cudnnDestroyTensorDescriptor(m_CUDNNBatchSummaryDesc));
        m_CUDNNBatchSummaryDesc = NULL;

        delete m_aTenTotalMean;
        delete m_aTenTotalVariance;
        delete m_aTenCachedMean;
        delete m_aTenCachedInvVariance;
#endif  // if __CUDNN__
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        DTYPE *CUDNNX = m_pTenInput->GetGPUData(pTime);

        DTYPE *CUDNNBnScale = m_pTenScale->GetGPUData(0);
        DTYPE *CUDNNBnBias  = m_pTenBias->GetGPUData(0);

        DTYPE *CUDNNY = m_pTenResult->GetGPUData(pTime);

        DTYPE *CUDNNTotalMean     = m_aTenTotalMean->GetGPUData(0);
        DTYPE *CUDNNTotalVariance = m_aTenTotalVariance->GetGPUData(0);

        DTYPE *CUDNNCachedMean        = NULL;
        DTYPE *CUDNNCachedInvVariance = NULL;

        float temp = 0.f;

        switch (m_mode) {
            case TRAINING:
                // CUDNNCachedMean        = m_aTenCachedMean->GetGPUData(0);
                // CUDNNCachedInvVariance = m_aTenCachedInvVariance->GetGPUData(0);
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               this->GetCudnnHandle(), m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               0.f, NULL, NULL,  /* CUDNNTotalMean, CUDNNTotalVariance,*/
                               m_CUDNNEpsilon, NULL, NULL  /* CUDNNCachedMean, CUDNNCachedInvVariance*/));
                break;
            case ACCUMULATING:
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               this->GetCudnnHandle(), m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor, CUDNNTotalMean, CUDNNTotalVariance,
                               m_CUDNNEpsilon, NULL, NULL)
                           );
                m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));
                break;
            case INFERENCING:
                checkCUDNN(cudnnBatchNormalizationForwardInference(
                               this->GetCudnnHandle(), m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               CUDNNTotalMean, CUDNNTotalVariance, m_CUDNNEpsilon));
                break;
            default:
                break;
        }
        checkCudaErrors(cudaDeviceSynchronize());


        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        DTYPE *CUDNNX       = m_pTenInput->GetGPUData(pTime);
        DTYPE *CUDNNBnScale = m_pTenScale->GetGPUData(0);
        DTYPE *CUDNNBnBias  = m_pTenBias->GetGPUData(0);
        DTYPE *CUDNNDx      = m_pTenDerInput->GetGPUData(pTime);
        DTYPE *CUDNNDy      = m_pTenDerResult->GetGPUData(pTime);

        DTYPE *CUDNNCachedMean        = m_aTenCachedMean->GetGPUData(0);
        DTYPE *CUDNNCachedInvVariance = m_aTenCachedInvVariance->GetGPUData(0);

        DTYPE *CUDNNBnScaleDiff = m_pTenDerScale->GetGPUData(0);
        DTYPE *CUDNNBnBiasDiff  = m_pTenDerBias->GetGPUData(0);

        checkCUDNN(cudnnBatchNormalizationBackward(
                       this->GetCudnnHandle(), m_CUDNNMode,
                       &m_CUDNNAlpha, &m_CUDNNBeta, &m_CUDNNAlpha, &m_CUDNNBeta,
                       m_CUDNNXDesc, CUDNNX, m_CUDNNDyDesc, CUDNNDy, m_CUDNNDxDesc, CUDNNDx,
                       m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnScaleDiff, CUDNNBnBiasDiff,
                       m_CUDNNEpsilon, NULL, NULL  /* CUDNNCachedMean, CUDNNCachedInvVariance*/));

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__

    int SetModeTraining() {
        if (m_mode == ACCUMULATING) {
            ;
        } else if (m_mode == INFERENCING) {
            ;
        } else {
            return TRUE;
        }
        m_mode = TRAINING;

        return TRUE;
    }

    int SetModeAccumulating() {
        if (m_mode == TRAINING) {
#ifdef __CUDNN__
            m_CUDNNExponentialAverageFactor = 1.0;
#endif  // ifdef __CUDNN__
            m_aTenTotalMean->Reset();
            m_aTenTotalVariance->Reset();
        } else if (m_mode == INFERENCING) {
            ;
        } else {
            return TRUE;
        }
        m_mode = ACCUMULATING;
        return TRUE;
    }

    int SetModeInferencing() {
#ifdef __CUDNN__
        // std::cout << "SetModeInferencing() : " << m_CUDNNExponentialAverageFactor << '\n';

        if (m_CUDNNExponentialAverageFactor < 1.0) {
            if (m_mode == TRAINING) {
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
#endif  // ifdef __CUDNN__

        m_mode = INFERENCING;
        return TRUE;
    }
};

#endif  // __CUDNN_BATCH_NORMALIZE__
