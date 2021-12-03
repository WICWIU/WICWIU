#ifndef FLIP_H_
#define FLIP_H_    value

#include "../Operator.hpp"


/*!
@class FlipTimeWise FlipTimeWise class
*/
template<typename DTYPE>
class FlipTimeWise : public Operator<DTYPE>{
private:

public:
    /**
    * @brief FlipTimeWise의 생성자
    * @details 파라미터로 받은 pInput0로 Alloc한다.
    * @param pInput0 Timewise로 flip할 Operator
    * @param pName 사용자가 부여한 Operator이름.
    */
    FlipTimeWise(Operator<DTYPE> *pInput0, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput0, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "FlipTimeWise::FlipTimeWise(Operator *)" << '\n';
        #endif  // __DEBUG__

        this->Alloc(pInput0);
    }

    ~FlipTimeWise() {
        std::cout << "FlipTimeWise::~FlipTimeWise()" << '\n';
    }

    /**
     * @brief 파라미터로 받은 pInput0으로 맴버 변수들을 초기화 한다.
     * @details pInput의 Shape을 사용하여 output으로 Result와 Delta로 사용 할 Tensor의 Shape을 정의한다.
     * @param pInput0 Timewise로 flip할 Operator
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput0) {
        #ifdef __DEBUG__
        std::cout << "FlipTimeWise::Alloc(Operator *, Operator *)" << '\n';
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

    /**
     * @brief FlipTimeWise의 ForwardPropagate 메소드
     * @details input의 timewise flip 결과를 resutl에 넣는다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int ForwardPropagate(int pTime = 0) {

        if(pTime !=0)
          return TRUE;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize     = result->GetTimeSize();
        int batchsize    = result->GetBatchSize();
        int channelsize  = result->GetChannelSize();
        int rowsize      = result->GetRowSize();
        int colsize      = result->GetColSize();

        Shape *inputTenShape = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        for(int ti=0; ti < timesize; ti++){
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                = (*input)[Index5D(inputTenShape, timesize - ti - 1, ba, ch, ro, co)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    /**
     * @brief FlipTimeWise의 BackPropagate 메소드
     * @details this_delta의 값을 timewise flip하여 input_delta에 넣는다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *inputGradient  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *thisGradient = this->GetGradient();

        int timesize     = inputGradient->GetTimeSize();
        int batchsize    = inputGradient->GetBatchSize();
        int channelsize  = inputGradient->GetChannelSize();
        int rowsize      = inputGradient->GetRowSize();
        int colsize      = inputGradient->GetColSize();

        if(pTime != timesize-1)
          return TRUE;

        Shape *inputTenShape = inputGradient->GetShape();
        Shape *resultTenShape = thisGradient->GetShape();

        for(int ti=0; ti < timesize; ti++){
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*inputGradient)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                                = (*thisGradient)[Index5D(resultTenShape, timesize - ti - 1, ba, ch, ro, co)];
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

#endif  // FLIP_H_
