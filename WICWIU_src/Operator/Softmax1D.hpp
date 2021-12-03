#ifndef SOFTMAX1D_H_
#define SOFTMAX1D_H_ value

#include "../Operator.hpp"
/**
 * @brief Tensor의 한 축을 대상으로 Softmax 연산을 수행하는 Operator 클래스.
 */
template <typename DTYPE>
class Softmax1D : public Operator<DTYPE> {
private:
    DTYPE m_epsilon;
    ///< Softmax1D연산 중 더해지는 epsilon값.
    Tensor<DTYPE> *m_aSumTensor;
    ///< 입력 Tensor의 대상 축을 기준으로 한 합 Tensor
    Tensor<DTYPE> *m_aMaxTensor;
    ///< 입력 Tensor의 대상 축에서 Max 값을 저장하는 Tensor
    Tensor<DTYPE> *m_aDeltaOutput;
    ///< Back Propagate 수행 과정에서 필요한 임시 값을 저장하는 Tensor

    int m_dim;
    ///< Softmax 연산을 수행할 대상 축의 index.
#ifdef __CUDNN__

    cudnnTensorDescriptor_t m_aInOutTensorDesc, m_aDeltaTensorDesc;
    cudnnTensorDescriptor_t m_aMaxTensorDesc, m_aSumTensorDesc;

    cudnnReduceTensorDescriptor_t m_aMaxReduceDesc, m_aNorm1ReduceDesc, m_aSumReduceDesc;
    cudnnOpTensorDescriptor_t m_aOpTensorDesc;


    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevSum, *m_pDevMax;
    DTYPE *m_pDevInputDelta, *m_pDevDelta, *m_pDevDeltaOutput;

    void *m_aDevMaxIndices, *m_aDevMaxWorkspace;
    void *m_aDevNorm1Indices, *m_aDevNorm1Workspace;
    void *m_aDevSumIndices, *m_aDevSumWorkspace;

    size_t m_MaxIndicesSizeInBytes, m_Norm1IndicesSizeInBytes, m_SumIndicesSizeInBytes;
    size_t m_MaxWorkspaceSizeInBytes, m_Norm1WorkspaceSizeInBytes, m_SumWorkspaceSizeInBytes;

    float m_alpha;
    float m_beta;

#endif
public:
    /*!
    @brief Softmax1D의 생성자.
    @details 파라미터로 받은 pOperator, epsilon을 Alloc시킨다.
    @param pOperator Softmax1D할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @param epsilon ForwardPropagate에 사용힐 값. 0으로 나누어지는 것을 방지하는 역할을 한다.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax1D(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f, int dim = 4, std::string pName = "no name", int pLoadflag = TRUE) : Operator<DTYPE>(pOperator, pName, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "Softmax1D::Softmax1D(Operator *)" << '\n';
#endif // __DEBUG__
        Alloc(pOperator, epsilon, dim);
    }
    /*!
    @brief Softmax1D의 소멸자.
    */
    ~Softmax1D() {
#ifdef __DEBUG__
        std::cout << "Softmax1D::~Softmax1D()" << '\n';
#endif // __DEBUG__
    }

    /*!
    @brief 파라미터로 받은 pOperator로 맴버변수들을 초기화 하고 Result, Gradient를 설정한다.
    @details input으로 받은 Operator의 Shape정보들로 맴버 변수드을 초기화 하고, 같은 Shape을 갖는 Tensor를 만들어 Result와 Gradient로 설정한다.
    @param pOperator Softmax1D할 Operator들
    @param epsilon 0으로 나누어지는 것을 방지하기위해 Softmax1D식의 분모에 더하는 값.
    @return 성공 시 TRUE.
    */
    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f, int dim = 4) {

        if (dim > 4 || dim < -5) {
            std::cout << "Out of Range" << '\n';
            return FALSE;
        }
        if (dim < 0) {
            dim = dim + 5;
        }
        m_epsilon = epsilon;

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int newDim[5] = { timesize, batchsize, channelsize, rowsize, colsize };
        newDim[dim]   = 1;
        m_dim         = dim;

        m_aSumTensor = Tensor<DTYPE>::Zeros(newDim[0], newDim[1], newDim[2], newDim[3], newDim[4]);
        m_aMaxTensor = Tensor<DTYPE>::Zeros(newDim[0], newDim[1], newDim[2], newDim[3], newDim[4]);

        m_aDeltaOutput = Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize);

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Shape *shape        = this->GetInput()[0]->GetResult()->GetShape();
        Shape *reducedShape = m_aSumTensor->GetShape();

        int timesize    = (*shape)[0];
        int batchsize   = (*shape)[1];
        int channelsize = (*shape)[2];
        int rowsize     = (*shape)[3];
        int colsize     = (*shape)[4];

        int reducedTimesize    = (*reducedShape)[0];
        int reducedBatchsize   = (*reducedShape)[1];
        int reducedChannelsize = (*reducedShape)[2];
        int reducedRowsize     = (*reducedShape)[3];
        int reducedColsize     = (*reducedShape)[4];

        m_alpha = 1.0f;
        m_beta  = 0.0f;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInOutTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aMaxTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aSumTensorDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aMaxTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              reducedBatchsize, reducedChannelsize, reducedRowsize, reducedColsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aSumTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              reducedBatchsize, reducedChannelsize, reducedRowsize, reducedColsize));

        checkCUDNN(cudnnCreateOpTensorDescriptor(&m_aOpTensorDesc));
        checkCUDNN(cudnnSetOpTensorDescriptor(m_aOpTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));


        checkCUDNN(cudnnCreateReduceTensorDescriptor(&m_aMaxReduceDesc));
        checkCUDNN(cudnnCreateReduceTensorDescriptor(&m_aNorm1ReduceDesc));
        checkCUDNN(cudnnCreateReduceTensorDescriptor(&m_aSumReduceDesc));

        checkCUDNN(cudnnSetReduceTensorDescriptor(m_aMaxReduceDesc, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES));
        checkCUDNN(cudnnSetReduceTensorDescriptor(m_aNorm1ReduceDesc, CUDNN_REDUCE_TENSOR_NORM1, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES));
        checkCUDNN(cudnnSetReduceTensorDescriptor(m_aSumReduceDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES));


        m_MaxIndicesSizeInBytes = 0;
        m_Norm1IndicesSizeInBytes = 0;
        m_MaxWorkspaceSizeInBytes = 0;
        m_Norm1WorkspaceSizeInBytes = 0;
        m_SumIndicesSizeInBytes = 0;
        m_SumWorkspaceSizeInBytes = 0;

        checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), m_aMaxReduceDesc, m_aInOutTensorDesc, m_aMaxTensorDesc, &m_MaxIndicesSizeInBytes));
        checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), m_aNorm1ReduceDesc, m_aInOutTensorDesc, m_aSumTensorDesc, &m_Norm1IndicesSizeInBytes));
        checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), m_aSumReduceDesc, m_aInOutTensorDesc, m_aSumTensorDesc, &m_SumIndicesSizeInBytes));

        checkCUDNN(cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), m_aMaxReduceDesc, m_aInOutTensorDesc, m_aMaxTensorDesc, &m_MaxWorkspaceSizeInBytes));
        checkCUDNN(cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), m_aNorm1ReduceDesc, m_aInOutTensorDesc, m_aSumTensorDesc, &m_Norm1WorkspaceSizeInBytes));
        checkCUDNN(cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), m_aSumReduceDesc, m_aInOutTensorDesc, m_aSumTensorDesc, &m_SumWorkspaceSizeInBytes));

        checkCudaErrors(cudaMalloc(&m_aDevMaxIndices, m_MaxIndicesSizeInBytes));
        checkCudaErrors(cudaMalloc(&m_aDevMaxWorkspace, m_MaxWorkspaceSizeInBytes));

        checkCudaErrors(cudaMalloc(&m_aDevNorm1Indices, m_Norm1IndicesSizeInBytes));
        checkCudaErrors(cudaMalloc(&m_aDevNorm1Workspace, m_Norm1WorkspaceSizeInBytes));

        checkCudaErrors(cudaMalloc(&m_aDevSumIndices, m_SumIndicesSizeInBytes));
        checkCudaErrors(cudaMalloc(&m_aDevSumWorkspace, m_SumWorkspaceSizeInBytes));

        m_aSumTensor->SetDeviceGPU(this->GetDeviceID());
        m_aMaxTensor->SetDeviceGPU(this->GetDeviceID());
        m_aDeltaOutput->SetDeviceGPU(this->GetDeviceID());

    }
#endif

    /*!
    @brief Alloc매소드에서 할당했던 sum, max를 삭제하고 포인터를 NULL로 초기화 한다.
    */
    virtual void Delete() {
        if (m_aSumTensor) {
            delete m_aSumTensor;
        }
        if (m_aMaxTensor) {
            delete m_aMaxTensor;
        }
        if (m_aDeltaOutput) {
            delete m_aDeltaOutput;
        }
#ifdef __CUDNN__

        if (m_aInOutTensorDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aInOutTensorDesc));
        m_aInOutTensorDesc = NULL;

        if (m_aDeltaTensorDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaTensorDesc));
        m_aDeltaTensorDesc = NULL;

        if (m_aMaxTensorDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aMaxTensorDesc));
        m_aMaxTensorDesc = NULL;

        if (m_aSumTensorDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aSumTensorDesc));
        m_aSumTensorDesc = NULL;

        if (m_aMaxReduceDesc)
            checkCUDNN(cudnnDestroyReduceTensorDescriptor(m_aMaxReduceDesc));
        m_aMaxReduceDesc = NULL;

        if (m_aNorm1ReduceDesc)
            checkCUDNN(cudnnDestroyReduceTensorDescriptor(m_aNorm1ReduceDesc));
        m_aNorm1ReduceDesc = NULL;

        if (m_aSumReduceDesc)
            checkCUDNN(cudnnDestroyReduceTensorDescriptor(m_aSumReduceDesc));
        m_aSumReduceDesc = NULL;

        if (m_aOpTensorDesc)
            checkCUDNN(cudnnDestroyOpTensorDescriptor(m_aOpTensorDesc));
        m_aOpTensorDesc = NULL;

        if (m_aDevMaxIndices)
            checkCudaErrors(cudaFree(m_aDevMaxIndices))
        m_aDevMaxIndices = NULL;
        if (m_aDevMaxWorkspace)
            checkCudaErrors(cudaFree(m_aDevMaxWorkspace))
        m_aDevMaxWorkspace = NULL;
        if (m_aDevNorm1Indices)
            checkCudaErrors(cudaFree(m_aDevNorm1Indices))
        m_aDevNorm1Indices = NULL;
        if (m_aDevNorm1Workspace)
            checkCudaErrors(cudaFree(m_aDevNorm1Workspace))
        m_aDevNorm1Workspace = NULL;
        if (m_aDevSumIndices)
            checkCudaErrors(cudaFree(m_aDevSumIndices))
        m_aDevSumIndices = NULL;
        if (m_aDevSumWorkspace)
            checkCudaErrors(cudaFree(m_aDevSumWorkspace))
        m_aDevSumWorkspace = NULL;


#endif
    }

    /*!
    @brief Softmax1D의 ForwardPropagate 매소드
    @details max값을 계산하고, exp()한 모든 값들을 더해 sum을 구한 뒤, 각각의 exp(input)한 값을 sum으로 나누어주어 확률값을 구하고 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input       = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result      = this->GetResult();
        Shape         *inputShape  = input->GetShape();
        Shape         *resultShape = result->GetShape();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        // (e^(x-max)+epsilon)/sum
        m_aMaxTensor->Reset();
        m_aSumTensor->Reset();
        Tensor<DTYPE>::GetMaxTensor(input, m_aMaxTensor, m_dim);

        Tensor<DTYPE> *exponential = new Tensor<DTYPE>(1, batchsize, channelsize, rowsize, colsize);
        Shape         *expShape    = exponential->GetShape();

        Shape *maxShape = m_aMaxTensor->GetShape();

        int compressedTi = (*maxShape)[0] - 1;
        int compressedBa = (*maxShape)[1] - 1;
        int compressedCh = (*maxShape)[2] - 1;
        int compressedRo = (*maxShape)[3] - 1;
        int compressedCo = (*maxShape)[4] - 1;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*exponential)[Index5D(expShape, 0, ba, ch, ro, co)] = exp((*input)[Index5D(inputShape, ti, ba, ch, ro, co)] - (*m_aMaxTensor)[Index5D(maxShape, MIN(ti, compressedTi), MIN(ba, compressedBa), MIN(ch, compressedCh), MIN(ro, compressedRo), MIN(co, compressedCo))]) + m_epsilon;
                    }
                }
            }
        }
        Tensor<DTYPE>::GetSumTensor(exponential, m_aSumTensor, m_dim);
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int index           = Index5D(inputShape, ti, ba, ch, ro, co);
                        int compressedIndex = Index5D(maxShape, MIN(ti, compressedTi), MIN(ba, compressedBa), MIN(ch, compressedCh), MIN(ro, compressedRo), MIN(co, compressedCo));
                        (*result)[index]    = (exp((*input)[index] - (*m_aMaxTensor)[compressedIndex]) + m_epsilon);
                        float temp          = (*result)[index];
                        (*result)[index] /= ((*m_aSumTensor)[compressedIndex] + m_epsilon);
                    }
                }
            }
        }

        delete exponential;
        return TRUE;
    }

    /*!
    @brief Softmax1D의 BackPropagate 매소드.
    @details Softmax1D의 미분 값을 구하여 input_delta에 넣어준다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int ti = pTime;

        Shape *inputShape = this_delta->GetShape();

        if (m_dim == 1) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int   index = Index5D(inputShape, ti, ba, ch, ro, co);
                            DTYPE temp  = (DTYPE)0;
                            for (int hid = 0; hid < batchsize; hid++) {
                                int tempIndex = Index5D(inputShape, ti, hid, ch, ro, co);
                                if (ba == hid)
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (1 - (*result)[tempIndex]);
                                else
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (-1 * (*result)[tempIndex]);
                            }
                            (*input_delta)[index] += temp;
                        }
                    }
                }
            }
        }
        else if (m_dim == 2) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int   index = Index5D(inputShape, ti, ba, ch, ro, co);
                            DTYPE temp  = (DTYPE)0;
                            for (int hid = 0; hid < channelsize; hid++) {
                                int tempIndex = Index5D(inputShape, ti, ba, hid, ro, co);
                                if (ch == hid)
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (1 - (*result)[tempIndex]);
                                else
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (-1 * (*result)[tempIndex]);
                            }
                            (*input_delta)[index] += temp;
                        }
                    }
                }
            }
        }
        else if (m_dim == 3) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int   index = Index5D(inputShape, ti, ba, ch, ro, co);
                            DTYPE temp  = (DTYPE)0;
                            for (int hid = 0; hid < rowsize; hid++) {
                                int tempIndex = Index5D(inputShape, ti, ba, ch, hid, co);
                                if (ro == hid)
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (1 - (*result)[tempIndex]);
                                else
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (-1 * (*result)[tempIndex]);
                            }
                            (*input_delta)[index] += temp;
                        }
                    }
                }
            }
        }
        else if (m_dim == 4) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int   index = Index5D(inputShape, ti, ba, ch, ro, co);
                            DTYPE temp  = (DTYPE)0;
                            for (int hid = 0; hid < colsize; hid++) {
                                int tempIndex = Index5D(inputShape, ti, ba, ch, ro, hid);
                                if (co == hid)
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (1 - (*result)[tempIndex]);
                                else
                                    temp += (*this_delta)[tempIndex] * (*result)[index] * (-1 * (*result)[tempIndex]);
                            }
                            (*input_delta)[index] += temp;
                        }
                    }
                }
            }
        }
        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__

};

#endif // Softmax1D_H_
