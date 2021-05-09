#ifndef HADAMARD_H_
#define HADAMARD_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Hadamard : public Operator<DTYPE>{
private:

public:
    Hadamard(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput0, pInput1, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Hadamard::Hadamard(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput0, pInput1);
    }


    ~Hadamard() {
        std::cout << "Hadamard::~Hadamard()" << '\n';
    }

    /*!
    @brief 파라미터로 받은 pInput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        #ifdef __DEBUG__
        std::cout << "Hadamard::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput0->GetResult()->GetTimeSize();
        int batchsize   = pInput0->GetResult()->GetBatchSize();
        int channelsize = pInput0->GetResult()->GetChannelSize();
        int rowsize     = pInput0->GetResult()->GetRowSize();
        int colsize     = pInput0->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }


    /*!
    @brief Hadamard의 ForwardPropagate 매소드
    @details input의 Tensor값들을 Hadamard을 취한 뒤 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input0  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input1  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = ((*input0)[Index5D(resultTenShape, ti, ba, ch, ro, co)]) *
                               ((*input1)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief Hadamard의 BackPropagate 매소드.
    @details result값으로 Hadamard의 미분 값을 계산하여 input_delta에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *input0      = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input1      = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta       = this->GetDelta();
        Tensor<DTYPE> *input0_delta  = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *input1_delta     = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        Shape *resultTenShape = this_delta->GetShape();
        Shape *input0TenShape = input0_delta->GetShape();
        Shape *input1TenShape = input1_delta->GetShape();

        int input0_index = 0;
        int input1_index  = 0;
        int result_index = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                      input0_index = Index5D(input0TenShape, ti, ba, ch, ro, co);
                      input1_index  = Index5D(input1TenShape, ti, ba, ch, ro, co);
                      result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                      (*input0_delta)[input0_index]      += (*input1)[input1_index] * (*this_delta)[result_index];
                      (*input1_delta)[input1_index]      += (*input0)[input0_index] * (*this_delta)[result_index];

                    }
                }
            }
        }

        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {

      return TRUE;
    }

#endif  // __CUDNN__
};

#endif  // HADAMARD_H_
