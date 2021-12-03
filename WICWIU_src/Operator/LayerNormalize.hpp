#ifndef LAYER_NORMALIZE_H
#define LAYER_NORMALIZE_H value

#include "../Operator.hpp"
/**
 * @brief Input Tensor에 대해 Layer Normalize를 수행하는 Operator Class.
 */
template <typename DTYPE> class LayerNormalize : public Operator<DTYPE> {
private:
    Tensor<DTYPE> *m_pMeanTensor;
    ///> 입력 Tensor의 한 축에 대한 평균을 저장하는 Tensor
    Tensor<DTYPE> *m_pVarTensor;
    ///> 입력 Tensor의 한 축에 대한 분산을 저장하는 Tensor
    Tensor<DTYPE> *m_pCachedNormalizedTensor;
    ///> 입력 Tensor의 한 축에 대한 정규화된 결과를 저장하는 Tensor

    int   m_batchIndex;
    ///> Batch로 사용하고 있는 축의 Index
    int   m_unbiased;
    ///> 표본평균과 표본분산을 사용할지에 대한 Flag
    float m_epsilon;
    ///> 정규화 과정에서 더해지는 Epsilon 값

#ifdef __CUDNN__

    Tensor<DTYPE> *m_pSqrtOfSquaredSumTensor;
    Tensor<DTYPE> *m_pMeanSquaredTensor;
    Tensor<DTYPE> *m_pXHat_delta;
    Tensor<DTYPE> *m_pXHatAvg_delta;
    Tensor<DTYPE> *m_pXHatScaled_delta;
    Tensor<DTYPE> *m_pXHatScaledAvg_delta;
    cudnnTensorDescriptor_t inputTenDesc;

    cudnnTensorDescriptor_t meanTenDesc, sqrtOfMeanSquaredTenDesc, meanSquaredTenDesc, varTenDesc;

    cudnnOpTensorDescriptor_t meanSquaredOpTenDesc;

    cudnnReduceTensorDescriptor_t meanReduceTenDesc;
    cudnnReduceTensorDescriptor_t sqrtOfMeanSquaredReduceTenSesc;

    DTYPE *m_pDevInput,  *m_pDevOutput, *m_pDevScale, *m_pDevBias,
          *m_pDevCachedNormalized, *m_pDevMean, *m_pDevSqrtOfSquaredSum, *m_pDevMeanSquared, *m_pDevVar, 
          *m_pDevInputDelta, *m_pDevDelta, *m_pDevScaleDelta, *m_pDevBiasDelta,
          *m_pDevXHatDelta, *m_pDevXHatScaleDelta, *m_pDevXHatAvgDelta, *m_pDevXHatScaleAvgDelta;

    DTYPE m_alpha;
    DTYPE m_beta;

    size_t meanWorkSpaceSize;
    size_t meanIndicesSize;

    size_t sqrtOfMeanSquaredWorkSpaceSize;
    size_t sqrtOfMeanSquareIndicesSize;

    void *meanWorkSpace;
    void *meanIndicesSpace;

    void *sqrtOfMeanSquaredWorkSpace;
    void *sqrtOfMeanSquareIndicesSpace;

#endif // __CUDNN__

public:
    /**
     * @brief LayerNormalize의 생성자.
     * @details 파라미터로 받은 pInput, pScale, pBias, batchIndex, unbiased, epsilon으로 Alloc 한다.
     * @param pInput Alloc할 대상 Operator
     * @param pScale Alloc할 대상 Operator
     * @param pBias Alloc할 대상 Operator
     * @param batchIndex Batch로 사용하고 있는 축의 Index, default값은 1
     * @param unbiased 표본평균, 표본표준편차를 사용할지에 대한 Flag, default값은 사용
     * @param epsilon 정규화 과정에서 더해지는 Epsilon 값
     */
    LayerNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int batchIndex = 1, int unbiased = TRUE,
                   float epsilon = 0.00001, std::string pName = "No Name", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pScale, pBias, pName, pLoadflag) {
#if __DEBUG__
        std::cout << "LayerNormalize<DTYPE>::LayerNormalize(Operator< DTYPE>*, Operator< DTYPE>*,  Operator< DTYPE>*, float, std:: string)" << '\n';
#endif // __DEBUG__

        Alloc(pInput, pScale, pBias, batchIndex, unbiased, epsilon);
    }

    /**
     * @brief 파라미터로 받은 pInput, pScale, pBias, batchIndex, unbiased, epsilon으로 Alloc 한다.
     * @details  Result와 Gradient를 저장하기 위해 pInput의 Shape에서 BatchIndex이후의 축들의 크기를 1로 바꾼다.
     * @param pInput 입력 Tensor를 가진 Operator
     * @param pScale Affine Transformation을 위한 Weight 역할을 하는 Operator
     * @param pBias Affine Transformation을 위한 Bias 역할을 하는 Operator
     * @param batchIndex Batch로 사용하고 있는 축의 Index
     * @param unbiased 표본평균, 표본표준편차를 사용할지에 대한 Flag, default값은 사용
     * @param epsilon 정규화 과정에서 더해지는 Epsilon 값
     * @return 성공시 TRUE
     */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int batchIndex, int unbiased, float epsilon) {
#ifdef __DEBUG__
        std::cout << "LayerNormalize<DTYPE>::Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, float "
                     "epsilon, std::string pName)"
                  << '\n';
#endif // __DEBUG__
        if (batchIndex >= 4) {
            std::cout << "Error! Layer should have bigger than 1 dimension" << '\n';
            return FALSE;
        }
        m_batchIndex = batchIndex;
        m_unbiased   = unbiased;
        m_epsilon    = epsilon;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int dims[5] = { timesize, batchsize, channelsize, rowsize, colsize };
        for (int i = m_batchIndex + 1; i < 5; i++) {
            dims[i] = 1;
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        m_pMeanTensor             = new Tensor<DTYPE>(dims[0], dims[1], dims[2], dims[3], dims[4]);
        m_pVarTensor              = new Tensor<DTYPE>(dims[0], dims[1], dims[2], dims[3], dims[4]);
        m_pCachedNormalizedTensor = new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize);
        m_pMeanTensor->Reset();
        m_pVarTensor->Reset();
        m_pCachedNormalizedTensor->Reset();

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {

        Operator<DTYPE> *pInput = this->GetInput()[0];
        Operator<DTYPE> *pScale = this->GetInput()[1];
        Operator<DTYPE> *pBias  = this->GetInput()[2];

        Shape *inputShape = pInput->GetResult()->GetShape();
        Shape *meanShape  = m_pMeanTensor->GetShape();

        int inputTimeSize    = (*inputShape)[0];
        int inputBatchSize   = (*inputShape)[1];
        int inputChannelSize = (*inputShape)[2];
        int inputRowSize     = (*inputShape)[3];
        int inputColSize     = (*inputShape)[4];

        int meanTimeSize    = (*meanShape)[0];
        int meanBatchSize   = (*meanShape)[1];
        int meanChannelSize = (*meanShape)[2];
        int meanRowSize     = (*meanShape)[3];
        int meanColSize     = (*meanShape)[4];

        m_pSqrtOfSquaredSumTensor = Tensor<DTYPE>::Zeros(meanTimeSize, meanBatchSize, meanChannelSize, meanRowSize,
                                                          meanColSize /*, "LayerNormalize_SqrtOfMeanSquaredTen"*/);
        m_pMeanSquaredTensor      = Tensor<DTYPE>::Zeros(meanTimeSize, meanBatchSize, meanChannelSize, meanRowSize,
                                                    meanColSize /*, "LayerNormalize_MeanSquaredTen"*/);

        m_pXHat_delta          = new Tensor<DTYPE>(new Shape(inputShape));
        m_pXHatAvg_delta       = new Tensor<DTYPE>(new Shape(meanShape));
        m_pXHatScaled_delta    = new Tensor<DTYPE>(new Shape(inputShape));
        m_pXHatScaledAvg_delta = new Tensor<DTYPE>(new Shape(meanShape));
    
        m_pXHat_delta->Reset();
        m_pXHatAvg_delta->Reset();
        m_pXHatScaled_delta->Reset();
        m_pXHatScaledAvg_delta->Reset();

        m_pSqrtOfSquaredSumTensor->SetDeviceGPU(this->GetDeviceID());
        m_pMeanSquaredTensor->SetDeviceGPU(this->GetDeviceID());
        m_pMeanTensor->SetDeviceGPU(this->GetDeviceID());
        m_pVarTensor->SetDeviceGPU(this->GetDeviceID());
        m_pCachedNormalizedTensor->SetDeviceGPU(this->GetDeviceID());
        m_pXHat_delta->SetDeviceGPU(this->GetDeviceID());
        m_pXHatAvg_delta->SetDeviceGPU(this->GetDeviceID());
        m_pXHatScaled_delta->SetDeviceGPU(this->GetDeviceID());
        m_pXHatScaledAvg_delta->SetDeviceGPU(this->GetDeviceID());
        

        m_alpha = 1;
        m_beta  = 0;

        meanWorkSpaceSize = 0;
        meanIndicesSize   = 0;

        sqrtOfMeanSquaredWorkSpaceSize = 0;
        sqrtOfMeanSquareIndicesSize    = 0;

        meanWorkSpace    = NULL;
        meanIndicesSpace = NULL;

        sqrtOfMeanSquaredWorkSpace   = NULL;
        sqrtOfMeanSquareIndicesSpace = NULL;

        checkCUDNN(cudnnCreateTensorDescriptor(&inputTenDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&sqrtOfMeanSquaredTenDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&meanSquaredTenDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&meanTenDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&varTenDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTenDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchSize, inputChannelSize,
                                              inputRowSize, inputColSize));

        checkCUDNN(cudnnSetTensor4dDescriptor(sqrtOfMeanSquaredTenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, meanBatchSize, meanChannelSize,
                                              meanRowSize, meanColSize));

        checkCUDNN(cudnnSetTensor4dDescriptor(meanSquaredTenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, meanBatchSize, meanChannelSize,
                                              meanRowSize, meanColSize));

        checkCUDNN(cudnnSetTensor4dDescriptor(meanTenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, meanBatchSize, meanChannelSize, meanRowSize,
                                              meanColSize));

        checkCUDNN(cudnnSetTensor4dDescriptor(varTenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, meanBatchSize, meanChannelSize, meanRowSize,
                                              meanColSize));

        checkCUDNN(cudnnCreateOpTensorDescriptor(&meanSquaredOpTenDesc));
        checkCUDNN(cudnnCreateReduceTensorDescriptor(&meanReduceTenDesc));
        checkCUDNN(cudnnCreateReduceTensorDescriptor(&sqrtOfMeanSquaredReduceTenSesc));

        checkCUDNN(cudnnSetOpTensorDescriptor(meanSquaredOpTenDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

        checkCUDNN(cudnnSetReduceTensorDescriptor(meanReduceTenDesc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                                  CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES)); // 8bit check
        checkCUDNN(cudnnSetReduceTensorDescriptor(sqrtOfMeanSquaredReduceTenSesc, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT,
                                                  CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES));

        checkCUDNN(
          cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), meanReduceTenDesc, inputTenDesc, meanTenDesc, &meanWorkSpaceSize));
        checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), meanReduceTenDesc, inputTenDesc, meanTenDesc, &meanIndicesSize));

        checkCUDNN(cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), sqrtOfMeanSquaredReduceTenSesc, inputTenDesc, varTenDesc,
                                                  &sqrtOfMeanSquaredWorkSpaceSize));
        checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), sqrtOfMeanSquaredReduceTenSesc, inputTenDesc, varTenDesc,
                                                &sqrtOfMeanSquareIndicesSize));

        if (meanWorkSpaceSize != 0) {
            checkCudaErrors(cudaMalloc(&meanWorkSpace, meanWorkSpaceSize));

            if (meanWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (meanIndicesSize != 0) {
            checkCudaErrors(cudaMalloc(&meanIndicesSpace, meanIndicesSize));

            if (meanIndicesSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (sqrtOfMeanSquaredWorkSpaceSize != 0) {
            checkCudaErrors(cudaMalloc(&sqrtOfMeanSquaredWorkSpace, sqrtOfMeanSquaredWorkSpaceSize));

            if (sqrtOfMeanSquaredWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (sqrtOfMeanSquareIndicesSize != 0) {
            checkCudaErrors(cudaMalloc(&sqrtOfMeanSquareIndicesSpace, sqrtOfMeanSquareIndicesSize));

            if (sqrtOfMeanSquareIndicesSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }
    }

#endif

    virtual void Delete() {
#ifdef __DEBUG__
        std::cout << "LayerNormalize<DTYPE>::Delete()" << '\n';
#endif // __DEBUG__
        if (m_pMeanTensor)
            delete m_pMeanTensor;
        if (m_pVarTensor)
            delete m_pVarTensor;
        if (m_pCachedNormalizedTensor)
            delete m_pCachedNormalizedTensor;

#ifdef __CUDNN__
        if (inputTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(inputTenDesc))
        inputTenDesc = NULL;

        if (sqrtOfMeanSquaredTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(sqrtOfMeanSquaredTenDesc))
        sqrtOfMeanSquaredTenDesc = NULL;

        if (meanSquaredTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(meanSquaredTenDesc))
        meanSquaredTenDesc = NULL;

        if (meanTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(meanTenDesc))
        meanTenDesc = NULL;

        if (varTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(varTenDesc))
        varTenDesc = NULL;

        if (inputTenDesc) 
            checkCUDNN(cudnnDestroyTensorDescriptor(inputTenDesc))
        inputTenDesc = NULL;

        if (meanReduceTenDesc)
            checkCUDNN(cudnnDestroyReduceTensorDescriptor(meanReduceTenDesc));
        meanReduceTenDesc = NULL;

        if (sqrtOfMeanSquaredReduceTenSesc)
            checkCUDNN(cudnnDestroyReduceTensorDescriptor(sqrtOfMeanSquaredReduceTenSesc));
        sqrtOfMeanSquaredReduceTenSesc = NULL;

        if (meanSquaredOpTenDesc)
            checkCUDNN(cudnnDestroyOpTensorDescriptor(meanSquaredOpTenDesc));
        meanSquaredOpTenDesc = NULL;

        if (meanWorkSpace)
            checkCudaErrors(cudaFree(meanWorkSpace));
        meanWorkSpace = NULL;

        if (meanIndicesSpace)
            checkCudaErrors(cudaFree(meanIndicesSpace));
        meanIndicesSpace = NULL;

        if (sqrtOfMeanSquaredWorkSpace)
            checkCudaErrors(cudaFree(sqrtOfMeanSquaredWorkSpace));
        sqrtOfMeanSquaredWorkSpace = NULL;

        if (sqrtOfMeanSquareIndicesSpace)
            checkCudaErrors(cudaFree(sqrtOfMeanSquareIndicesSpace));
        sqrtOfMeanSquareIndicesSpace = NULL;

        if (m_pSqrtOfSquaredSumTensor)
            delete m_pSqrtOfSquaredSumTensor;
        m_pSqrtOfSquaredSumTensor = NULL;

        if (m_pMeanSquaredTensor)
            delete m_pMeanSquaredTensor;
        m_pMeanSquaredTensor = NULL;

        if (m_pXHat_delta)
            delete m_pXHat_delta;
        m_pXHat_delta = NULL;

        if (m_pXHatAvg_delta)
            delete m_pXHatAvg_delta;
        m_pXHatAvg_delta = NULL;

        if (m_pXHatScaled_delta)
            delete m_pXHatScaled_delta;
        m_pXHatScaled_delta = NULL;

        if (m_pXHatScaledAvg_delta)
            delete m_pXHatScaledAvg_delta;
        m_pXHatScaledAvg_delta = NULL;

#endif
    }

    /**
     * @brief LayerNormalize의 ForwardPropagate 메소드.
     * @details input Tensor의 평균과 분산을 계산하고, 정규화 연산을 수행한뒤, Affine Transformation을 수행한다.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int ForwardPropagate(int pTime = 0) {
        //평균, 분산 최신화
        ComputeLayerStatistics();
        //정규화
        Normalize(pTime);
        // ScaleAndShift
        AffineTransform(pTime);
        return TRUE;
    }

    /**
     * @brief LayerNormalize의 BackPropagate 메소드.
     * @details Scale과 Bias Parameter의 Gradient를 계산하고, 정규화 연산에 대한 Gradient를 계산한다.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE
     */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();

        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = input_container->GetElement(0)->GetDelta();
        Tensor<DTYPE> *scale_delta = input_container->GetElement(1)->GetDelta();
        Tensor<DTYPE> *bias_delta  = input_container->GetElement(2)->GetDelta();

        Tensor<DTYPE> *scale_result = input_container->GetElement(1)->GetResult();

        Shape *pShape             = this_delta->GetShape();
        Shape *pAffineShape       = scale_delta->GetShape();
        Shape *pLayerSummaryShape = m_pMeanTensor->GetShape();

        Tensor<DTYPE> *xhat_delta             = new Tensor<DTYPE>(new Shape(pShape));
        Tensor<DTYPE> *xhatSummed_delta       = new Tensor<DTYPE>(new Shape(pLayerSummaryShape));
        Tensor<DTYPE> *xhatScaled_delta       = new Tensor<DTYPE>(new Shape(pShape));
        Tensor<DTYPE> *xhatScaledSummed_delta = new Tensor<DTYPE>(new Shape(pLayerSummaryShape));

        xhat_delta->Reset();
        xhatSummed_delta->Reset();
        xhatScaled_delta->Reset();
        xhatScaledSummed_delta->Reset();

        int ti          = pTime;
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        int n = 1;

        if (m_batchIndex == 0) {
            n = batchsize * channelsize * rowsize * colsize;
        }
        else if (m_batchIndex == 1) {
            n = channelsize * rowsize * colsize;
        }
        else if (m_batchIndex == 2) {
            n = rowsize * colsize;
        }
        else if (m_batchIndex == 3) {
            n = colsize;
        }

        // Scale & Bias BackPropagation
        if (m_batchIndex == 3) {
            for (int co = 0; co < colsize; co++) {
                int   affineIndex = Index5D(pAffineShape, 0, 0, 0, 0, co);
                DTYPE biasDelta   = 0;
                DTYPE scaleDelta  = 0;
                for (int ba = 0; ba < batchsize; ba++) {
                    for (int ch = 0; ch < channelsize; ch++) {
                        for (int ro = 0; ro < rowsize; ro++) {
                            int index = Index5D(pShape, ti, ba, ch, ro, co);
                            biasDelta += (*this_delta)[index];
                            scaleDelta += (*this_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                            (*xhat_delta)[index] += (*this_delta)[index] * (*scale_result)[affineIndex];
                            (*xhatScaled_delta)[index] += (*xhat_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                        }
                    }
                }
                (*bias_delta)[affineIndex]  = biasDelta;
                (*scale_delta)[affineIndex] = scaleDelta;
            }
        }
        else if (m_batchIndex == 2) {
            for (int co = 0; co < colsize; co++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    int   affineIndex = Index5D(pAffineShape, 0, 0, 0, ro, co);
                    DTYPE biasDelta   = 0;
                    DTYPE scaleDelta  = 0;
                    for (int ba = 0; ba < batchsize; ba++) {
                        for (int ch = 0; ch < channelsize; ch++) {
                            int index = Index5D(pShape, ti, ba, ch, ro, co);
                            biasDelta += (*this_delta)[index];
                            scaleDelta += (*this_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                            (*xhat_delta)[index] += (*this_delta)[index] * (*scale_result)[affineIndex];
                            (*xhatScaled_delta)[index] += (*xhat_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                        }
                    }
                    (*bias_delta)[affineIndex]  = biasDelta;
                    (*scale_delta)[affineIndex] = scaleDelta;
                }
            }
        }
        else if (m_batchIndex == 1) {
            for (int co = 0; co < colsize; co++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int ch = 0; ch < channelsize; ch++) {
                        int   affineIndex = Index5D(pAffineShape, 0, 0, ch, ro, co);
                        DTYPE biasDelta   = 0;
                        DTYPE scaleDelta  = 0;
                        for (int ba = 0; ba < batchsize; ba++) {
                            int index = Index5D(pShape, ti, ba, ch, ro, co);
                            biasDelta += (*this_delta)[index];
                            scaleDelta += (*this_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                            (*xhat_delta)[index] += (*this_delta)[index] * (*scale_result)[affineIndex];
                            (*xhatScaled_delta)[index] += (*xhat_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                        }
                        (*bias_delta)[affineIndex]  = biasDelta;
                        (*scale_delta)[affineIndex] = scaleDelta;
                    }
                }
            }
        }
        else if (m_batchIndex == 0) {
            for (int co = 0; co < colsize; co++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int ch = 0; ch < channelsize; ch++) {
                        for (int ba = 0; ba < batchsize; ba++) {
                            int   affineIndex = Index5D(pAffineShape, 0, ba, ch, ro, co);
                            DTYPE biasDelta   = 0;
                            DTYPE scaleDelta  = 0;
                            int   index       = Index5D(pShape, ti, ba, ch, ro, co);
                            biasDelta += (*this_delta)[index];
                            scaleDelta += (*this_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                            (*xhat_delta)[index] += (*this_delta)[index] * (*scale_result)[affineIndex];
                            (*xhatScaled_delta)[index] += (*xhat_delta)[index] * (*m_pCachedNormalizedTensor)[index];
                            (*bias_delta)[affineIndex]  = biasDelta;
                            (*scale_delta)[affineIndex] = scaleDelta;
                        }
                    }
                }
            }
        }

        // Normalize BackPropagation
        if (m_batchIndex == 0) {
            Tensor<DTYPE>::GetSumTensorOverAxes(xhat_delta, xhatSummed_delta, 4, 1, 2, 3, 4);
            Tensor<DTYPE>::GetSumTensorOverAxes(xhatScaled_delta, xhatScaledSummed_delta, 4, 1, 2, 3, 4);
        }
        else if (m_batchIndex == 1) {
            Tensor<DTYPE>::GetSumTensorOverAxes(xhat_delta, xhatSummed_delta, 3, 2, 3, 4);
            Tensor<DTYPE>::GetSumTensorOverAxes(xhatScaled_delta, xhatScaledSummed_delta, 3, 2, 3, 4);
        }
        else if (m_batchIndex == 2) {
            Tensor<DTYPE>::GetSumTensorOverAxes(xhat_delta, xhatSummed_delta, 2, 3, 4);
            Tensor<DTYPE>::GetSumTensorOverAxes(xhatScaled_delta, xhatScaledSummed_delta, 2, 3, 4);
        }
        else if (m_batchIndex == 3) {
            Tensor<DTYPE>::GetSumTensorOverAxes(xhat_delta, xhatSummed_delta, 1, 4);
            Tensor<DTYPE>::GetSumTensorOverAxes(xhatScaled_delta, xhatScaledSummed_delta, 1, 4);
        }

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        unsigned int index        = Index5D(pShape, ti, ba, ch, ro, co);
                        unsigned int summaryIndex = GetLayerSummaryIndex(ti, ba, ch, ro, co);
                        (*input_delta)[index] += 1.f / (n * sqrt((*m_pVarTensor)[summaryIndex] + m_epsilon)) *
                                                 (n * (*xhat_delta)[index] - (*xhatSummed_delta)[summaryIndex] -
                                                  ((*xhatScaledSummed_delta)[summaryIndex] * (*m_pCachedNormalizedTensor)[index]));
                    }
                }
            }
        }

        delete xhat_delta;
        delete xhatSummed_delta;
        delete xhatScaled_delta;
        delete xhatScaledSummed_delta;
        return TRUE;
    }

    /**
     * @brief Input Tensor의 평균과 분산을 계산하는 메소드.
     * @details Input Tensor의 평균과 분산을 제공된 Parameter들을 적절히 제공하여 TensorMath의 메소드들을 활용하여 계산.
     * @see TensorMath
     */
    void ComputeLayerStatistics() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        if (m_batchIndex == 0) {
            Tensor<DTYPE>::GetVarTensorOverAxes(input, m_pVarTensor, m_unbiased, 4, 1, 2, 3, 4);
            Tensor<DTYPE>::GetMeanTensorOverAxes(input, m_pMeanTensor, 4, 1, 2, 3, 4);
        }
        else if (m_batchIndex == 1) {
            Tensor<DTYPE>::GetVarTensorOverAxes(input, m_pVarTensor, m_unbiased, 3, 2, 3, 4);
            Tensor<DTYPE>::GetMeanTensorOverAxes(input, m_pMeanTensor, 3, 2, 3, 4);
        }
        else if (m_batchIndex == 2) {
            Tensor<DTYPE>::GetVarTensorOverAxes(input, m_pVarTensor, m_unbiased, 2, 3, 4);
            Tensor<DTYPE>::GetMeanTensorOverAxes(input, m_pMeanTensor, 2, 3, 4);
        }
        else if (m_batchIndex == 3) {
            Tensor<DTYPE>::GetVarTensorOverAxes(input, m_pVarTensor, m_unbiased, 1, 4);
            Tensor<DTYPE>::GetMeanTensorOverAxes(input, m_pMeanTensor, 1, 4);
        }
    }

    /**
     * @brief Input Tensor의 평균과 분산을 바탕으로 Input Tensor의 정규화된 값을 Cache에 저장하는 메소드.
     * @details 앞서 계산한 평균과 분산을 활용하여 결과값 계산.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @see ComputeLayerStatistics
     */
    void Normalize(int pTime) {

        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        int            ti          = pTime;
        int            batchsize   = input->GetBatchSize();
        int            channelsize = input->GetChannelSize();
        int            rowsize     = input->GetRowSize();
        int            colsize     = input->GetColSize();

        Shape *pShape             = this->GetResult()->GetShape();
        Shape *pLayerSummaryShape = m_pMeanTensor->GetShape();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        unsigned int index        = Index5D(pShape, ti, ba, ch, ro, co);
                        unsigned int summaryIndex = GetLayerSummaryIndex(ti, ba, ch, ro, co);
                        (*m_pCachedNormalizedTensor)[index] =
                          ((*input)[index] - (*m_pMeanTensor)[summaryIndex]) / sqrt(m_epsilon + (*m_pVarTensor)[summaryIndex]);
                    }
                }
            }
        }
    }

    /**
     * @brief 정규화된 Input Tensor에 Affine Transformation을 적용하는 메소드.
     * @details 정규화된 값을 학습 가능한 Weight과 Bias를 통해 Transformation 수행.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @see ComputeLayerStatistics, Normalize
     */
    void AffineTransform(int pTime) {
        Operator<DTYPE> *pInput      = this->GetInput()[0];
        int              ti          = pTime;
        int              batchsize   = pInput->GetResult()->GetBatchSize();
        int              channelsize = pInput->GetResult()->GetChannelSize();
        int              rowsize     = pInput->GetResult()->GetRowSize();
        int              colsize     = pInput->GetResult()->GetColSize();
        Tensor<DTYPE> *  result      = this->GetResult();
        Shape *          pShape      = result->GetShape();

        Tensor<DTYPE> *scale        = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *bias         = this->GetInput()[2]->GetResult();
        Shape *        pAffineShape = scale->GetShape();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        unsigned int index       = Index5D(pShape, ti, ba, ch, ro, co);
                        unsigned int affineIndex = Index5D(pAffineShape, 0, 0, 0, 0, co);
                        (*result)[index] += (*m_pCachedNormalizedTensor)[index] * (*scale)[affineIndex] + (*bias)[affineIndex];
                    }
                }
            }
        }
    }

    /**
     * @brief input Tensor의 index를 기반으로 압축된 Index를 계산하는 메소드. 
     * @param timeIndex Input Tensor의 Time 축 Index
     * @param batchIndex Input Tensor의 Batch 축 Index
     * @param channelIndex Input Tensor의 Channel 축 Index
     * @param rowIndex Input Tensor의 Row 축 Index
     * @param colIndex Input Tensor의 Column 축 Index
     * @return 압축된 Index
     */
    unsigned int GetLayerSummaryIndex(int timeIndex, int batchIndex, int channelIndex, int rowIndex, int colIndex) {
        unsigned int index = 0;
        if (m_batchIndex == 0) {
            index = Index5D(m_pMeanTensor->GetShape(), timeIndex, 0, 0, 0, 0);
        }
        else if (m_batchIndex == 1) {
            index = Index5D(m_pMeanTensor->GetShape(), timeIndex, batchIndex, 0, 0, 0);
        }
        else if (m_batchIndex == 2) {
            index = Index5D(m_pMeanTensor->GetShape(), timeIndex, batchIndex, channelIndex, 0, 0);
        }
        else if (m_batchIndex == 3) {
            index = Index5D(m_pMeanTensor->GetShape(), timeIndex, batchIndex, channelIndex, rowIndex, 0);
        }

        return index;
    }
    Tensor<DTYPE> *GetLayerMean() { return m_pMeanTensor; }
    Tensor<DTYPE> *GetLayerVar() { return m_pVarTensor; }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);
    int BackPropagateOnGPU(int pTime);

    int ComputeLayerStatisticsOnGPU();
    int NormalizeOnGPU();
    int AffineTransformOnGPU();
#endif // __CUDNN__
};

#endif // LAYER_NORMALIZE_H
