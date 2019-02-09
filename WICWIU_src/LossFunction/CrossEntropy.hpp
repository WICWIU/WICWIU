#ifndef CROSSENTROPY_H_
#define CROSSENTROPY_H_    value

#include "../LossFunction.hpp"

/*!
@class CrossEntropy Cross Entropy Metric를 이용해 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details Cross Entropy 계산식을 이용해 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와 레이블 값의 손실 함수를 계산한다
*/
template<typename DTYPE>
class CrossEntropy : public LossFunction<DTYPE>{
private:
    DTYPE m_epsilon = 0.0;  // for backprop

public:
    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 epsilon을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param epsilon 더미 변수, 값을 미 지정시 1e-6f로 초기화
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon = 1e-6f) : LossFunction<DTYPE>(pOperator, pLabel) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 1e-6f에 해당하는 epsilon 값을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param pName LossFunction의 이름
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, 1e-6f);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 epsilon을 매개변수로 전달하여 CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드를 호출한다.
    @param pOperator CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon) 메소드의 매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param epsilon 더미 변수
    @param pName LossFunction의 이름
    @return 없음
    @see CrossEntropy<DTYPE>::Alloc(Operator<DTYPE> *pOperator, int epsilon)
    */
    CrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }

    /*!
    @brief CrossEntropy LossFunction 클래스 소멸자
    @return 없음
    */
    ~CrossEntropy() {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::~CrossEntropy()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief CrossEntropy Lossfunction의 멤버 변수들을 동적 할당하는 메소드
    @details 매개변수로 전달받은 Operator를 Input Operator에 할당하고 초기화 된 Result 텐서를 동적으로 할당 및 생성한다.
    @param pOperator CrossEntropy LossFunction의 입력에 해당하는 Operator
    @param epsilon 더미 변수
    @return TRUE
    */
    virtual int Alloc(Operator<DTYPE> *pOperator, int epsilon) {
        #ifdef __DEBUG__
        std::cout << "CrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    /*!
    @brief CrossEntropy LossFunction의 순전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 결과 값을 레이블 값과 비교해 Cross Entropy를 구한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return 뉴럴 네트워크의 결과 값에 대한 Cross Entropy
    */
    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = (ti * batchsize + ba);

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += -(*label)[index] * log((*input)[index] + m_epsilon);
            }
        }

        return result;
    }

    /*!
    @brief CrossEntropy LossFunction의 역전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 CrossEntropy LossFunction에 대한 입력 Tensor의 Gradient를 계산한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return NULL
    */
    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[index] += -(*label)[index] / (*input)[index];
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    /*!
    @brief GPU 동작 모드에서의 CrossEntropy LossFunction의 순전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>CrossEntropy::ForwardPropagate(int pTime = 0)
    */
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;
    }

    /*!
    @brief GPU 동작 모드에서의 CrossEntropy LossFunction의 역전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>CrossEntropy::BackPropagate(int pTime = 0)
    */
    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif  // __CUDNN__
};

#endif  // CROSSENTROPY_H_
