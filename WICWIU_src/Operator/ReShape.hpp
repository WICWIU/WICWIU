#ifndef RESHAPE_H_
#define RESHAPE_H_ value

#include "../Operator.hpp"

template <typename DTYPE>
class ReShape : public Operator<DTYPE>
{
public:
    /*!
    @brief ReShape의 생성자
    @details 파라미터로 받은 pInput, pRowSize, pColSize으로 Alloc한다.
    @param pInput ReShape할 Operator.
    @param pRowSize ReShape으로 새로 만들어질 Tensor의 rowsize.
    @param pColSize ReShape으로 새로 만들어질 Tensor의 colsize.
    @paramp pName 사용자가 부여한 Operator의 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int
    pRowSize, int pColSize)
    */
    ReShape(Operator<DTYPE>* pInput, int pRowSize, int pColSize, std::string pName,
            int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
#endif // __DEBUG__
        this->Alloc(pInput, 0, 0, 0, pRowSize, pColSize);
    }

    /*!
    @brief ReShape의 생성자
    @details 파라미터로 받은 pInput, pRowSize, pColSize으로 Alloc한다.
    @param pInput ReShape할 Operator.
    @param pChannelSize ReShape으로 새로 만들어질 Tensor의 channelsize
    @param pRowSize ReShape으로 새로 만들어질 Tensor의 rowsize.
    @param pColSize ReShape으로 새로 만들어질 Tensor의 colsize.
    @paramp pName 사용자가 부여한 Operator의 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int
    pRowSize, int pColSize)
    */
    ReShape(Operator<DTYPE>* pInput, int pChannelSize, int pRowSize, int pColSize,
            std::string pName, int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
#endif // __DEBUG__
        this->Alloc(pInput, 0, 0, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief ReShape의 생성자
    @details 파라미터로 받은 pInput, pRowSize, pColSize으로 Alloc한다.
    @param pInput ReShape할 Operator.
    @param pBatchSize ReShape으로 새로 만들어질 Tensor의 batchsize.
    @param pChannelSize ReShape으로 새로 만들어질 Tensor의 channelsize.
    @param pRowSize ReShape으로 새로 만들어질 Tensor의 rowsize.
    @param pColSize ReShape으로 새로 만들어질 Tensor의 colsize.
    @paramp pName 사용자가 부여한 Operator의 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int
    pRowSize, int pColSize)
    */
    ReShape(Operator<DTYPE>* pInput, int pBatchSize, int pChannelSize, int pRowSize, int pColSize,
            std::string pName, int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
#endif // __DEBUG__
        this->Alloc(pInput, 0, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief ReShape의 생성자
    @details 파라미터로 받은 pInput, pRowSize, pColSize으로 Alloc한다.
    @param pInput ReShape할 Operator.
    @param pTimeSize ReShape으로 새로 만들어질 Tensor의 timesize.
    @param pBatchSize ReShape으로 새로 만들어질 Tensor의 batchsize.
    @param pChannelSize ReShape으로 새로 만들어질 Tensor의 channelsize.
    @param pRowSize ReShape으로 새로 만들어질 Tensor의 rowsize.
    @param pColSize ReShape으로 새로 만들어질 Tensor의 colsize.
    @paramp pName 사용자가 부여한 Operator의 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int
    pRowSize, int pColSize)
    */
    ReShape(Operator<DTYPE>* pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize,
            int pColSize, std::string pName, int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "ReShape::ReShape(Operator *)" << '\n';
#endif // __DEBUG__
        this->Alloc(pInput, pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    /*!
    @brief ReShape의 소멸자.
    @details Delete 매소드를 사용한다.
    @ref void Delete()
    */
    ~ReShape()
    {
#ifdef __DEBUG__
        std::cout << "ReShape::~ReShape()" << '\n';
#endif // __DEBUG__

        Delete();
    }

    /*!
    @brief 파라미터로 받은 값들로 Shape의 dim들을 바꾼다.
    @details result에 파라미터로 받은 값들로 result의 shape을 바꾸어 넣고,
    @details Delta도 마찬가지로 받은 Dimension 정보들로 새로운 Tensor를 생성하여 넣는다,
    @param pInput ReShape할 Operator.
    @param pTimeSize ReShape으로 새로 만들어질 Tensor의 timesize.
    @param pBatchSize ReShape으로 새로 만들어질 Tensor의 batchsize.
    @param pChannelSize ReShape으로 새로 만들어질 Tensor의 channelsize.
    @param pRowSize ReShape으로 새로 만들어질 Tensor의 rowsize.
    @param pColSize ReShape으로 새로 만들어질 Tensor의 colsize.
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE>* pInput, int pTimeSize, int pBatchSize, int pChannelSize,
              int pRowSize, int pColSize)
    {
#ifdef __DEBUG__
        std::cout << "ReShape::Alloc(Operator *, Operator *)" << '\n';
#endif // __DEBUG__

        Shape* pInputShape = pInput->GetResult()->GetShape();

        if (pTimeSize == 0)
            pTimeSize = (*pInputShape)[0];

        if (pBatchSize == 0)
            pBatchSize = (*pInputShape)[1];

        if (pChannelSize == 0)
            pChannelSize = (*pInputShape)[2];

        Tensor<DTYPE>* result = new Tensor<DTYPE>(pInput->GetResult());
        result->ReShape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        this->SetResult(result); // copy data

        this->SetDelta(new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));

        return TRUE;
    }

    /*!
    @brief  Delete 매소드.
    @details 별다른 기능은 없다.
    */
    void Delete() {}

    /*!
    @brief ReShape의 ForwardPropagate 매소드
    @details input값을 result(새로운 Shape을 갖는 Tensor)에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        int timesize = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize = result->GetRowSize();
        int colsize = result->GetColSize();

        Shape* resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] =
                            (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief ReShape의 BackPropagate 매소드.
    @details input_delta에 this_delta값을 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* this_delta = this->GetDelta();
        Tensor<DTYPE>* input_delta = this->GetInput()[0]->GetDelta();

        int timesize = this_delta->GetTimeSize();
        int batchsize = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize = this_delta->GetRowSize();
        int colsize = this_delta->GetColSize();

        Shape* deltaTenShape = this_delta->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        (*input_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)] +=
                            (*this_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 ReShape의 ForwardPropagate 메소드.
    @details cudnnAddTensor를 이용해 pDevInput의 값을 pDevResult에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime)
    {
        Tensor<DTYPE>* input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        DTYPE* pDevInput = input->GetGPUData();
        DTYPE* pDevResult = result->GetGPUData();

        cudnnTensorDescriptor_t pDesc = input->GetDescriptor();

        float alpha = 1.f;
        float beta = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevInput, &alpha, pDesc,
                                  pDevResult));

        // this->ForwardPropagate(pTime);
        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 ReShape의 BackPropagateOnGPU 메소드.
    @details cudnnAddTensor를 이용해 pDevDelta의 값을 pDevInputDelta에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime)
    {
        Tensor<DTYPE>* this_delta = this->GetDelta();
        Tensor<DTYPE>* input_delta = this->GetInput()[0]->GetDelta();

        DTYPE* pDevDelta = this_delta->GetGPUData();
        DTYPE* pDevInputDelta = input_delta->GetGPUData();

        cudnnTensorDescriptor_t pDesc = this_delta->GetDescriptor();

        float alpha = 1.f;
        float beta = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevDelta, &alpha, pDesc,
                                  pDevInputDelta));

        // this->BackPropagate(pTime);

        return TRUE;
    }

#endif // __CUDNN__
};

#endif // RESHAPE_H_
