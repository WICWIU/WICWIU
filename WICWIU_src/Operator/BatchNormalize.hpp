#ifndef __CUDNN_BATCH_NORMALIZE__
#define __CUDNN_BATCH_NORMALIZE__    value

#include "../Operator.hpp"

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
    float m_momentum;
    double m_exponentialAverageFactor;

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
    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise = TRUE, std::string pName = NULL, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pScale, pBias, pName, pLoadflag) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise);
    }

    BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise = TRUE, float pMomentum = 0.1, std::string pName = NULL, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pScale, pBias, pName, pLoadflag) {
#if __DEBUG__
        std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pScale, pBias, pIsChannelwise, pMomentum);
    }

    ~BatchNormalize() {
#if __DEBUG__
        std::cout << "BatchNormalize:: ~ BatchNormalize()" << '\n';
#endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pMomentum = 0.1, double pEpsilon = 0.01) {
#if __DEBUG__
        std::cout << "BatchNormalize:: Alloc( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, double)" << '\n';
#endif  // __DEBUG__

        m_pTenInput         = pInput->GetResult();
        m_aTenTotalVariance = pScale->GetResult();
        m_aTenTotalMean     = pBias->GetResult();

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
        m_momentum      = pMomentum;

        if (m_isChannelwise) {
            m_batchSummaryCapacity = m_numChannel;
            // m_aTenTotalMean         = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            // m_aTenTotalVariance     = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_pTenBias              = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_pTenScale             = Tensor<DTYPE>::Constants(1, 1, m_numChannel, 1, 1, 1);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, m_numChannel, 1, 1);
        } else {
            m_batchSummaryCapacity  = m_numInputColumn;
            m_pTenBias              = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_numInputColumn);
            m_pTenScale             = Tensor<DTYPE>::Constants(1, 1, 1, 1, m_numInputColumn, 1);
            m_aTenCachedMean        = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_numInputColumn);
            m_aTenCachedInvVariance = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_numInputColumn);
        }

        this->SetResult(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));
        this->SetGradient(new Tensor<DTYPE>(m_inputTimeSize, m_inputBatchSize, m_numChannel, m_numInputRow, m_numInputColumn));

        m_pTenResult    = this->GetResult();
        m_pTenDerResult = this->GetGradient();

        m_mode = TRAIN;

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        checkCudaErrors(cudaSetDevice(idOfDevice));
        m_pTenBias->SetDeviceGPU(idOfDevice);
        m_pTenScale->SetDeviceGPU(idOfDevice);

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

        // m_aTenTotalMean->SetDeviceGPU(idOfDevice);
        // m_aTenTotalVariance->SetDeviceGPU(idOfDevice);
        m_aTenCachedMean->SetDeviceGPU(idOfDevice);
        m_aTenCachedInvVariance->SetDeviceGPU(idOfDevice);

        m_CUDNNAlpha   = 1.f;
        m_CUDNNBeta    = 0.f;
        m_CUDNNEpsilon = CUDNN_BN_MIN_EPSILON;

        if (m_momentum != 0) m_CUDNNExponentialAverageFactor = m_momentum;
        else m_CUDNNExponentialAverageFactor = 1.0;
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
            case TRAIN:
                m_aTenCachedMean->Reset(m_CUDNNHandle);
                m_aTenCachedInvVariance->Reset(m_CUDNNHandle);
                CUDNNCachedMean        = m_aTenCachedMean->GetGPUData(0);
                CUDNNCachedInvVariance = m_aTenCachedInvVariance->GetGPUData(0);
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor, CUDNNTotalMean, CUDNNTotalVariance,
                               m_CUDNNEpsilon, CUDNNCachedMean, CUDNNCachedInvVariance));

                if (m_momentum == 0) m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));  // for exponential
                break;
            case ACCUMULATE:
                checkCUDNN(cudnnBatchNormalizationForwardTraining(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               m_CUDNNExponentialAverageFactor, CUDNNTotalMean, CUDNNTotalVariance,
                               m_CUDNNEpsilon, NULL, NULL)
                           );
                m_CUDNNExponentialAverageFactor = (m_CUDNNExponentialAverageFactor / (m_CUDNNExponentialAverageFactor + 1));
                break;
            case INFERENCE:
                checkCUDNN(cudnnBatchNormalizationForwardInference(
                               m_CUDNNHandle, m_CUDNNMode, &m_CUDNNAlpha, &m_CUDNNBeta,
                               m_CUDNNXDesc, CUDNNX, m_CUDNNYDesc, CUDNNY,
                               m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnBias,
                               CUDNNTotalMean, CUDNNTotalVariance, m_CUDNNEpsilon));
                break;
            default:
                break;
        }

        // checkCudaErrors(cudaDeviceSynchronize());


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
                       &m_CUDNNAlpha, &m_CUDNNAlpha, &m_CUDNNAlpha, &m_CUDNNAlpha,
                       m_CUDNNXDesc, CUDNNX, m_CUDNNDyDesc, CUDNNDy, m_CUDNNDxDesc, CUDNNDx,
                       m_CUDNNBatchSummaryDesc, CUDNNBnScale, CUDNNBnScaleDiff, CUDNNBnBiasDiff,
                       m_CUDNNEpsilon, CUDNNCachedMean, CUDNNCachedInvVariance  /* CUDNNCachedMean, CUDNNCachedInvVariance*/));


        m_pTenDerScale->Reset(m_CUDNNHandle);
        m_pTenDerBias->Reset(m_CUDNNHandle);

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__

    int SetModeTrain() {
        if (m_mode == ACCUMULATE) {
#ifdef __CUDNN__

            if (m_momentum == 0) m_CUDNNExponentialAverageFactor = 1.0;
            // m_aTenTotalMean->Reset(m_CUDNNHandle);
            // m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else if (m_mode == INFERENCE) {
#ifdef __CUDNN__

            if (m_momentum == 0) m_CUDNNExponentialAverageFactor = 1.0;
            // m_aTenTotalMean->Reset(m_CUDNNHandle);
            // m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else {
            return TRUE;
        }
        m_mode = TRAIN;

        return TRUE;
    }

    int SetModeAccumulate() {
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';

        if (m_mode == TRAIN) {
#ifdef __CUDNN__
            m_CUDNNExponentialAverageFactor = 1.0;
            // m_aTenTotalMean->Reset(m_CUDNNHandle);
            // m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else if (m_mode == INFERENCE) {
#ifdef __CUDNN__
            m_CUDNNExponentialAverageFactor = 1.0;
            // m_aTenTotalMean->Reset(m_CUDNNHandle);
            // m_aTenTotalVariance->Reset(m_CUDNNHandle);
#endif  // ifdef __CUDNN__
        } else {
            return TRUE;
        }
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';

        m_mode = ACCUMULATE;
        return TRUE;
    }

    int SetModeInference() {
        // std::cout << m_aTenTotalMean << '\n';
        // std::cout << m_aTenTotalVariance << '\n';
        if (m_mode == TRAIN) {
            ;
        } else if (m_mode == ACCUMULATE) {
            ;
        } else {
            return TRUE;
        }
        m_mode = INFERENCE;
        return TRUE;
    }
};

#endif  // __CUDNN_BATCH_NORMALIZE__
