#ifndef SOFTMAX_H_
#define SOFTMAX_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Softmax : public Operator<DTYPE>{
    DTYPE m_epsilon;
    ///< Softmax연산 중 더해지는 epsilon값.

    int m_timesize;
    ///< 연산 할 Tensor가 위치한 Time값.

    DTYPE **sum;
    ///< Softmax연산 중 Tensor값들의 합을 저장하기 위한 포인터.
    DTYPE **max;
    ///< Softmax연산 중 Tensor값들 중 가장 큰 값을 저장하기 위한 포인터.

public:
    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator, epsilon을 Alloc시킨다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @param epsilon ForwardPropagate에 사용힐 값. 0으로 나누어지는 것을 방지하는 역할을 한다.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f) : Operator<DTYPE>(pOperator) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator을 Alloc한다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @param pName 사용자가 Operator에 부여한 이름.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, std::string pName) : Operator<DTYPE>(pOperator, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator);
    }

    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator, epsilon을 Alloc시킨다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @prram epsilon ForwardPropagate에 사용힐 값. 0으로 나누어지는 것을 방지하는 역할을 한다.
    @param pName 사용자가 Operator에 부여한 이름.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pOperator, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    /*!
    @brief Softmax의 소멸자.
    */
    ~Softmax() {
        #ifdef __DEBUG__
        std::cout << "Softmax::~Softmax()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief 파라미터로 받은 pOperator로 맴버변수들을 초기화 하고 Result, Gradient를 설정한다.
    @details input으로 받은 Operator의 Shape정보들로 맴버 변수드을 초기화 하고, 같은 Shape을 갖는 Tensor를 만들어 Result와 Gradient로 설정한다.
    @param pOperator Softmax할 Operator들
    @param epsilon 0으로 나누어지는 것을 방지하기위해 softmax식의 분모에 더하는 값.
    @return 성공 시 TRUE.
    */
    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f) {
        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    /*!
    @brief Alloc매소드에서 할당했던 sum, max를 삭제하고 포인터를 NULL로 초기화 한다.
    */
    virtual void Delete() {
        if (sum) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
    }

    /*!
    @brief Softmax의 ForwardPropagate 매소드
    @details max값을 계산하고, exp()한 모든 값들을 더해 sum을 구한 뒤, 각각의 exp(input)한 값을 sum으로 나누어주어 확률값을 구하고 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {  // thread
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*result)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
            }
        }

        return TRUE;
    }

    /*!
    @brief softmax의 BackPropagate 매소드.
    @details softmax의 미분 값을 구하여 input_delta에 넣어준다.
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

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        float temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            temp = 0.f;

            for (int i = start; i < end; i++) {
                temp += (*this_delta)[i] * (*result)[i];
            }

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = (*result)[i] * ((*this_delta)[i] - temp);
            }
        }

        return TRUE;
    }

    /*!
    @brief 파라미터로 받은 Tensor에서 가장 큰 값을 반환하는 함수.
    @param input 가장 큰 값을 찾을 대상 Tensor.
    @param start 값을 찾을 Tensor안에서의 시작위치.
    @param end 값을 찾을 Tensor안에서의 종료위치.
    @return input Tensor의 값들 중 가장 큰 값..
    */
    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate(pTime);
        return TRUE;
    }

#endif  // if __CUDNN__
};

#endif  // SOFTMAX_H_
