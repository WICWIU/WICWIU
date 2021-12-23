#ifndef BROADMATMUL_H_
#define BROADMATMUL_H_    value

#include "../Operator.hpp"

/**
 * @brief 두 Tensor의 Broadcasting을 지원하는 행렬곱을 수행하는 Operator Class.
 */
template<typename DTYPE> class BroadMatMul : public Operator<DTYPE>{
private:
#ifdef __CUDNN__

    DTYPE *m_pDevOutput, *m_pDevLeft, *m_pDevRight,
          *m_pDevDelta, *m_pDevLeftDelta, *m_pDevRightDelta;

    int m_gemmLoopSize;
    int m_gemmBatchSize;
    int m_gemmRowSize;
    int m_gemmColSize;
    int m_gemmHidSize;

    int m_LeftMatrixStride;
    int m_RightMatrixStride;

    int m_BatchedLeftMatrixStride;
    int m_BatchedRightMatrixStride;

    DTYPE **m_aOutputList, **m_aLeftList, **m_aRightList,
          **m_aDeltaList, **m_aLeftDeltaList, **m_aRightDeltaList;

    DTYPE **m_aDevOutputList, **m_aDevLeftList, **m_aDevRightList,
          **m_aDevDeltaList, **m_aDevLeftDeltaList, **m_aDevRightDeltaList;

    DTYPE m_alpha, m_beta;
    DTYPE m_backAlpha, m_backBeta;
#endif

public:
    /**
     * @brief BroadMatMul의 생성자.
     * @details 파라미터로 받은 pLeft, pRight로 Alloc한다.
     * @param pLeft Alloc할 좌항 대상 Operator
     * @param pRight Alloc할 우항 대상 Operator
     */
    BroadMatMul(Operator<DTYPE> *pLeft, Operator<DTYPE> *pRight, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pLeft, pRight, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "BroadMatMul::BroadMatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pLeft, pRight);
    }

    /**
     * @brief BroadMatMul의 소멸자.
     * @details Delete매소드를 사용해 GPU에 할당했던 값들을 해제한다.
     * @see void Delete()
     */
    virtual ~BroadMatMul() {
        #ifdef __DEBUG__
        std::cout << "BroadMatMul::~BroadMatMul()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /**
     * @brief 파라미터로 받은 pLeft, pRight로 멤버 변수들을 초기화 한다.
     * @details pLeft와 pRight의 Row, Column 축의 크기는 행렬곱 정의에 부합해야한다.
     * @details 이외의 축에 대해서는, Left와 Right Operator들 중 큰 곳에 맞춰진다.
     * @param pLeft 행렬 곱의 좌항 Operator
     * @param pRight 행렬 곱의 좌항 Operator
     * @return 성공 시 TRUE
     */
    int Alloc(Operator<DTYPE> *pLeft, Operator<DTYPE> *pRight) {
        #ifdef __DEBUG__
        std::cout << "BroadMatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int dim[5] = {1, 1, 1, 1, 1};
        Shape *pLeftShape  = pLeft->GetResult()->GetShape();
        Shape *pRightShape = pRight->GetResult()->GetShape();

        for (int i = 2; i >= 0; i --) {
            if ((*pLeftShape)[i] == (*pRightShape)[i] || (*pLeftShape)[i] == 1 || (*pRightShape)[i] == 1) {
                dim[i] = std::max((*pLeftShape)[i], (*pRightShape)[i]);
            }
            else {
                std::cout << "Cannot Broadcast! Please Check Shape of Two Tensor!!" << '\n';
            }
        }

        if ((*pLeftShape)[4] != (*pRightShape)[3]) {
            std::cout << "Cannot Multiply! Please Check Shape of Two Tensor!!" << '\n';
        }

        int timesize    = dim[0];
        int batchsize   = dim[1];
        int channelsize = dim[2];
        int rowsize     = (*pLeftShape)[3];
        int colsize     = (*pRightShape)[4];

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_alpha = 1;
        m_beta = 0;
        m_backAlpha = 1;
        m_backBeta = 1;

        Shape *pLeftShape  = this->GetInput()[0]->GetResult()->GetShape();
        Shape *pRightShape = this->GetInput()[1]->GetResult()->GetShape();
        Shape *pShape = this->GetResult()->GetShape();

        m_gemmLoopSize  = 1;
        m_gemmBatchSize = 1;
        m_gemmRowSize   = 1;
        m_gemmColSize   = 1;
        m_gemmHidSize   = this->GetInput()[0]->GetResult()->GetColSize();

        m_LeftMatrixStride  = 1;
        m_RightMatrixStride = 1;

        m_BatchedLeftMatrixStride  = 1;
        m_BatchedRightMatrixStride = 1;

        int dimension = 0;
        for (int i = 4; i >= 1; i --) {
            if ((*pShape)[i] != 1) {
                dimension ++;
                switch(dimension) {
                    case 1:
                        m_gemmColSize = (*pShape)[i];
                        break;
                    case 2:
                        m_gemmRowSize = (*pShape)[i];
                        break;
                    case 3:
                        m_gemmBatchSize = (*pShape)[i];
                        if ((*pLeftShape)[i] == 1)
                            m_LeftMatrixStride = 0;
                        else
                            m_LeftMatrixStride  = m_gemmRowSize * m_gemmHidSize;
                        if ((*pRightShape)[i] == 1)
                            m_RightMatrixStride = 0;
                        else
                            m_RightMatrixStride = m_gemmHidSize * m_gemmColSize;
                        break;
                    case 4:
                        m_gemmLoopSize = (*pShape)[i];
                        if ((*pLeftShape)[i] == 1)
                            m_BatchedLeftMatrixStride = 0;
                        else
                            m_BatchedLeftMatrixStride  = m_gemmRowSize * m_gemmHidSize * (*pLeftShape)[i+1];
                        if ((*pRightShape)[i] == 1)
                            m_BatchedRightMatrixStride = 0;
                        else
                            m_BatchedRightMatrixStride = m_gemmHidSize * m_gemmColSize * (*pRightShape)[i+1];
                        break;
                }
            }
        }
        if (m_gemmLoopSize == 1) {
            m_BatchedLeftMatrixStride = 0;
            m_BatchedRightMatrixStride = 0;
        }

        m_aOutputList     = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aOutputList);
        m_aLeftList       = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aLeftList);
        m_aRightList      = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aRightList);
        m_aDeltaList      = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aDeltaList);
        m_aLeftDeltaList  = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aLeftDeltaList);
        m_aRightDeltaList = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aRightDeltaList);

        checkCudaErrors(cudaMalloc(&m_aDevOutputList,     m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevLeftList,       m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevRightList,      m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevDeltaList,      m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevLeftDeltaList,  m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevRightDeltaList, m_gemmBatchSize * sizeof(float *)));

    }

#endif  // if __CUDNN__

    /**
     * @brief GPU에 할당했던 메모리를 해제하고 각 포인터들을 NULL로 초기화한다.
     * @details m_aOutputList, m_aLeftList, m_aRightList, m_aDeltaList, m_aLeftDeltaList, m_aRightDeltaList
     * @details m_aDevOutputList, m_aDevLeftList, m_aDevRightList, m_aDevDeltaList, m_aDevLeftDeltaList, m_aDevRightDeltaList
     * @details 들을 삭제하고 NULL로 초기화한다.
     * @details cudnn연산을 위해 할당 했던 메모리들을 해제시킨다.
     */
    void Delete() {
#ifdef __CUDNN__
        if (m_aOutputList)
            free(m_aOutputList);
        m_aOutputList = NULL;

        if (m_aLeftList)
            free(m_aLeftList);
        m_aLeftList = NULL;

        if (m_aRightList)
            free(m_aRightList);
        m_aRightList = NULL;

        if (m_aDeltaList)
            free(m_aDeltaList);
         m_aDeltaList = NULL;

        if (m_aLeftDeltaList)
            free(m_aLeftDeltaList);
        m_aLeftDeltaList = NULL;

        if (m_aRightDeltaList)
            free(m_aRightDeltaList);
        m_aRightDeltaList = NULL;


        if (m_aDevOutputList)
            checkCudaErrors(cudaFree(m_aDevOutputList));
        m_aDevOutputList = NULL;

        if (m_aDevLeftList)
            checkCudaErrors(cudaFree(m_aDevLeftList));
        m_aDevLeftList = NULL;

        if (m_aDevRightList)
            checkCudaErrors(cudaFree(m_aDevRightList));
        m_aDevRightList = NULL;

        if (m_aDevDeltaList)
            checkCudaErrors(cudaFree(m_aDevDeltaList));
        m_aDevDeltaList = NULL;

        if (m_aDevLeftDeltaList)
            checkCudaErrors(cudaFree(m_aDevLeftDeltaList));
        m_aDevLeftDeltaList = NULL;

        if (m_aDevRightDeltaList)
            checkCudaErrors(cudaFree(m_aDevRightDeltaList));
        m_aDevRightDeltaList = NULL;
#endif  // if __CUDNN__
    }

    /**
     * @brief BraodMatMul의 ForwardPropagate 메소드.
     * @details 두 Tensor에 대해 행렬곱을 수행한다.
     * @details MIN 을 통해, Index Out of Range를 방지하면서 BroadCasting을 적용한다.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = pLeft->GetColSize();

        Shape *leftShape   = pLeft->GetShape();
        Shape *rightShape  = pRight->GetShape();
        Shape *resultShape = result->GetShape();

        int leftDim[5] = {(*leftShape)[0] - 1, (*leftShape)[1] - 1, (*leftShape)[2] - 1, (*leftShape)[3] - 1, (*leftShape)[4] - 1};
        int rightDim[5] = {(*rightShape)[0] - 1, (*rightShape)[1] - 1, (*rightShape)[2] - 1, (*rightShape)[3] - 1, (*rightShape)[4] - 1};

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            (*result)[Index5D(resultShape, ti, ba, ch, ro, co)]
                                += (*pLeft)[Index5D(leftShape, MIN(ti, leftDim[0]), MIN(ba, leftDim[1]), MIN(ch, leftDim[2]), ro, hid)]
                                   * (*pRight)[Index5D(rightShape, MIN(ti, rightDim[0]), MIN(ba, rightDim[1]), MIN(ch, rightDim[2]), hid, co)];
                        }
                    }
                }
            }
        }


        return TRUE;
    }

    /**
     * @brief BraodMatMul의 BackPropagate 메소드.
     * @details 두 Tensor에 대해 행렬곱의 역전파를 수행한다.
     * @details MIN 을 통해, Index Out of Range를 방지하면서 BroadCasting을 적용한다.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *pLeft = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *pRight  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *left_delta  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *right_delta = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        int hiddensize  = left_delta->GetColSize();

        Shape *leftShape   = left_delta->GetShape();
        Shape *rightShape  = right_delta->GetShape();
        Shape *resultShape = this_delta->GetShape();


        int leftDim[5] = {(*leftShape)[0] - 1, (*leftShape)[1] - 1, (*leftShape)[2] - 1, (*leftShape)[3] - 1, (*leftShape)[4] - 1};
        int rightDim[5] = {(*rightShape)[0] - 1, (*rightShape)[1] - 1, (*rightShape)[2] - 1, (*rightShape)[3] - 1, (*rightShape)[4] - 1};

        int left_index = 0;
        int right_index  = 0;
        int result_index = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            left_index   = Index5D(leftShape, MIN(ti, leftDim[0]), MIN(ba, leftDim[1]), MIN(ch, leftDim[2]), ro, hid);
                            right_index  = Index5D(rightShape, MIN(ti, rightDim[0]), MIN(ba, rightDim[1]), MIN(ch, rightDim[2]), hid, co);
                            result_index = Index5D(resultShape, ti, ba, ch, ro, co);

                            (*left_delta)[left_index]   += (*pRight)[right_index] * (*this_delta)[result_index];
                            (*right_delta)[right_index] += (*pLeft)[left_index] * (*this_delta)[result_index];
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
#endif
};

template<typename DTYPE> class BahdanauBroadMatMul : public Operator<DTYPE>{
private:
#ifdef __CUDNN__

    DTYPE *m_pDevOutput, *m_pDevLeft, *m_pDevRight,
          *m_pDevDelta, *m_pDevLeftDelta, *m_pDevRightDelta;

    int m_gemmLoopSize;
    int m_gemmBatchSize;
    int m_gemmRowSize;
    int m_gemmColSize;
    int m_gemmHidSize;

    int m_LeftMatrixStride;
    int m_RightMatrixStride;

    int m_BatchedLeftMatrixStride;
    int m_BatchedRightMatrixStride;

    DTYPE **m_aOutputList, **m_aLeftList, **m_aRightList,
          **m_aDeltaList, **m_aLeftDeltaList, **m_aRightDeltaList;

    DTYPE **m_aDevOutputList, **m_aDevLeftList, **m_aDevRightList,
          **m_aDevDeltaList, **m_aDevLeftDeltaList, **m_aDevRightDeltaList;

    DTYPE m_alpha, m_beta;
    DTYPE m_backAlpha, m_backBeta;
#endif

public:

    BahdanauBroadMatMul(Operator<DTYPE> *pLeft, Operator<DTYPE> *pRight, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pLeft, pRight, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "BahdanauBroadMatMul::BahdanauBroadMatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pLeft, pRight);
    }


    virtual ~BahdanauBroadMatMul() {
        #ifdef __DEBUG__
        std::cout << "BahdanauBroadMatMul::~BahdanauBroadMatMul()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pLeft, Operator<DTYPE> *pRight) {
        #ifdef __DEBUG__
        std::cout << "BahdanauBroadMatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int dim[5] = {1, 1, 1, 1, 1};
        Shape *pLeftShape  = pLeft->GetResult()->GetShape();
        Shape *pRightShape = pRight->GetResult()->GetShape();

        for (int i = 2; i >= 0; i --) {
            if ((*pLeftShape)[i] == (*pRightShape)[i] || (*pLeftShape)[i] == 1 || (*pRightShape)[i] == 1) {
                dim[i] = std::max((*pLeftShape)[i], (*pRightShape)[i]);
            }
            else {
                std::cout << "Cannot Broadcast! Please Check Shape of Two Tensor!!" << '\n';
            }
        }

        if ((*pLeftShape)[4] != (*pRightShape)[3]) {
            std::cout << "Cannot Multiply! Please Check Shape of Two Tensor!!" << '\n';
        }

        int timesize    = dim[0];
        int batchsize   = dim[1];
        int channelsize = dim[2];
        int rowsize     = (*pLeftShape)[3];
        int colsize     = (*pRightShape)[4];

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_alpha = 1;
        m_beta = 0;
        m_backAlpha = 1;
        m_backBeta = 1;

        Shape *pLeftShape  = this->GetInput()[0]->GetResult()->GetShape();
        Shape *pRightShape = this->GetInput()[1]->GetResult()->GetShape();
        Shape *pShape = this->GetResult()->GetShape();

        m_gemmLoopSize  = 1;
        m_gemmBatchSize = 1;
        m_gemmRowSize   = 1;
        m_gemmColSize   = 1;
        m_gemmHidSize   = this->GetInput()[0]->GetResult()->GetColSize();

        m_LeftMatrixStride  = 1;
        m_RightMatrixStride = 1;

        m_BatchedLeftMatrixStride  = 1;
        m_BatchedRightMatrixStride = 1;

        int dimension = 0;
        for (int i = 4; i >= 1; i --) {
            if ((*pShape)[i] != 1) {
                dimension ++;
                switch(dimension) {
                    case 1:
                        m_gemmColSize = (*pShape)[i];
                        break;
                    case 2:
                        m_gemmRowSize = (*pShape)[i];
                        break;
                    case 3:
                        m_gemmBatchSize = (*pShape)[i];
                        if ((*pLeftShape)[i] == 1)
                            m_LeftMatrixStride = 0;
                        else
                            m_LeftMatrixStride  = m_gemmRowSize * m_gemmHidSize;
                        if ((*pRightShape)[i] == 1)
                            m_RightMatrixStride = 0;
                        else
                            m_RightMatrixStride = m_gemmHidSize * m_gemmColSize;
                        break;
                    case 4:
                        m_gemmLoopSize = (*pShape)[i];
                        if ((*pLeftShape)[i] == 1)
                            m_BatchedLeftMatrixStride = 0;
                        else
                            m_BatchedLeftMatrixStride  = m_gemmRowSize * m_gemmHidSize * (*pLeftShape)[i+1];
                        if ((*pRightShape)[i] == 1)
                            m_BatchedRightMatrixStride = 0;
                        else
                            m_BatchedRightMatrixStride = m_gemmHidSize * m_gemmColSize * (*pRightShape)[i+1];
                        break;
                }
            }
        }
        if (m_gemmLoopSize == 1) {
            m_BatchedLeftMatrixStride = 0;
            m_BatchedRightMatrixStride = 0;
        }

        m_aOutputList     = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aOutputList);
        m_aLeftList       = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aLeftList);
        m_aRightList      = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aRightList);
        m_aDeltaList      = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aDeltaList);
        m_aLeftDeltaList  = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aLeftDeltaList);
        m_aRightDeltaList = (float **)malloc(m_gemmBatchSize * sizeof(float *)); assert(m_aRightDeltaList);

        checkCudaErrors(cudaMalloc(&m_aDevOutputList,     m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevLeftList,       m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevRightList,      m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevDeltaList,      m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevLeftDeltaList,  m_gemmBatchSize * sizeof(float *)));
        checkCudaErrors(cudaMalloc(&m_aDevRightDeltaList, m_gemmBatchSize * sizeof(float *)));

    }

#endif  // if __CUDNN__

    void Delete() {
#ifdef __CUDNN__
        if (m_aOutputList)
            free(m_aOutputList);
        m_aOutputList = NULL;

        if (m_aLeftList)
            free(m_aLeftList);
        m_aLeftList = NULL;

        if (m_aRightList)
            free(m_aRightList);
        m_aRightList = NULL;

        if (m_aDeltaList)
            free(m_aDeltaList);
         m_aDeltaList = NULL;

        if (m_aLeftDeltaList)
            free(m_aLeftDeltaList);
        m_aLeftDeltaList = NULL;

        if (m_aRightDeltaList)
            free(m_aRightDeltaList);
        m_aRightDeltaList = NULL;


        if (m_aDevOutputList)
            checkCudaErrors(cudaFree(m_aDevOutputList));
        m_aDevOutputList = NULL;

        if (m_aDevLeftList)
            checkCudaErrors(cudaFree(m_aDevLeftList));
        m_aDevLeftList = NULL;

        if (m_aDevRightList)
            checkCudaErrors(cudaFree(m_aDevRightList));
        m_aDevRightList = NULL;

        if (m_aDevDeltaList)
            checkCudaErrors(cudaFree(m_aDevDeltaList));
        m_aDevDeltaList = NULL;

        if (m_aDevLeftDeltaList)
            checkCudaErrors(cudaFree(m_aDevLeftDeltaList));
        m_aDevLeftDeltaList = NULL;

        if (m_aDevRightDeltaList)
            checkCudaErrors(cudaFree(m_aDevRightDeltaList));
        m_aDevRightDeltaList = NULL;
#endif  // if __CUDNN__
    }

    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = pLeft->GetColSize();

        Shape *leftShape   = pLeft->GetShape();
        Shape *rightShape  = pRight->GetShape();
        Shape *resultShape = result->GetShape();

        int leftDim[5] = {(*leftShape)[0] - 1, (*leftShape)[1] - 1, (*leftShape)[2] - 1, (*leftShape)[3] - 1, (*leftShape)[4] - 1};
        int rightDim[5] = {(*rightShape)[0] - 1, (*rightShape)[1] - 1, (*rightShape)[2] - 1, (*rightShape)[3] - 1, (*rightShape)[4] - 1};

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            (*result)[Index5D(resultShape, ti, ba, ch, ro, co)]
                                += (*pLeft)[Index5D(leftShape, MIN(ti, leftDim[0]), MIN(ba, leftDim[1]), MIN(ch, leftDim[2]), ro, hid)]
                                   * (*pRight)[Index5D(rightShape, MIN(ti, rightDim[0]), MIN(ba, rightDim[1]), MIN(ch, rightDim[2]), hid, co)];
                        }
                    }
                }
            }
        }


        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *pLeft = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *pRight  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *left_delta  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *right_delta = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        int hiddensize  = left_delta->GetColSize();

        Shape *leftShape   = left_delta->GetShape();
        Shape *rightShape  = right_delta->GetShape();
        Shape *resultShape = this_delta->GetShape();


        int leftDim[5] = {(*leftShape)[0] - 1, (*leftShape)[1] - 1, (*leftShape)[2] - 1, (*leftShape)[3] - 1, (*leftShape)[4] - 1};
        int rightDim[5] = {(*rightShape)[0] - 1, (*rightShape)[1] - 1, (*rightShape)[2] - 1, (*rightShape)[3] - 1, (*rightShape)[4] - 1};

        int left_index = 0;
        int right_index  = 0;
        int result_index = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            left_index   = Index5D(leftShape, MIN(ti, leftDim[0]), MIN(ba, leftDim[1]), MIN(ch, leftDim[2]), ro, hid);
                            right_index  = Index5D(rightShape, MIN(ti, rightDim[0]), MIN(ba, rightDim[1]), MIN(ch, rightDim[2]), hid, co);
                            result_index = Index5D(resultShape, ti, ba, ch, ro, co);

                            (*left_delta)[left_index]   += (*pRight)[right_index] * (*this_delta)[result_index];
                            (*right_delta)[right_index] += (*pLeft)[left_index] * (*this_delta)[result_index];
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
#endif
};


#endif
