#ifndef TENSORHOLDER_H_
#define TENSORHOLDER_H_ value

#include "../Operator.hpp"

/*!
@class Tensorholder Gradient없이 Operator의 Result만 사용하기 위한 클래스.
@details 주로 Network의 input, label값을 저장하기 위해 구현되었다.
*/
template <typename DTYPE>
class Tensorholder : public Operator<DTYPE>
{

public:
    /*!
    @brief Tensorholder의 생성자.
    @details 파라미터로 받은 pTensor, pTrainable으로 Alloc한다.
    @param pTensor Alloc에 사용할 Tensor, 결론적으로 Tensorholder의 Result로 설정된다.
    @param pName 사용자가 부여한 Tensorholder의 이름.
    @param pTrainable 생성 할 Operator(Tensorholder)가 Trainable인지 알리는 변수. default로 TRUE를
    사용한다.
    @ref int Alloc(Tensor<DTYPE> *pTensor, int pTrainable)
    */
    Tensorholder(Tensor<DTYPE>* pTensor, std::string pName, int pTrainable = TRUE,
                 int pLoadflag = TRUE)
        : Operator<DTYPE>(pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::Tensorholder(Tensor<DTYPE> *, std::string)" << '\n';
#endif // __DEBUG__
        this->Alloc(pTensor, pTrainable);
    }

    /*!
    @brief Tensorholder의 생성자.
    @details 파리미터로 받은 pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize, pTrainable로
    Alloc한다.
    @details 파라미터로 받은 변수들은 Alloc에서 생성 할 Tensor의 Shape을 결정한다.
    @param pTimeSize Alloc에 사용 할 timesize.
    @param pBatchSize Alloc에 사용 할 batchsize.
    @param pChannelSize Alloc에 사용 할 channelsize.
    @param pRowSize Alloc에 사용 할 rowsize
    @param pColSize Alloc에 사용 할 colsize
    @param pName 사용자가 부여한 Tensorholder의 이름.
    @param pTrainable 생성 할 Operator(Tensorholder)가 Trainable인지 알리는 변수. default로 TRUE를
    사용한다.
    @ref int Alloc(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, int
    pTrainable)
    */
    Tensorholder(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize,
                 std::string pName, int pTrainable = TRUE, int pLoadflag = TRUE)
        : Operator<DTYPE>(pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Placeholder<DTYPE>::Placeholder(int, int, int, int, int, std::string)"
                  << '\n';
#endif // __DEBUG__

        this->Alloc(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize, pTrainable);
    }

    /*!
    @brief Tensorholder의 소멸자.
    @details 딱히 하는 일은 없다.
    */
    ~Tensorholder()
    {
#ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::~Tensorholder()" << '\n';
#endif // __DEBUG__
    }

    /*!
    @brief 파라미터로 뱓은 pTensor로 Tensorholder를 설정한다.
    @details 파라미터로 받은 pTensor를 Result값으로 설정한다.
    @details SetIsTensorholder를 통해 Tensorholder임을 설정하고, pTensor의 Shape과 같은 Shape을 갖는
    Tensor를 만들어 Gradient로 설정한다.
    @param pTensor Tensorholder의 Result로 저장 될값을 가진 Tensor.
    @param pTrainable Training이 가능한지 아닌지 알리는 변수
    @return 성공 시 TRUE.
    @ref int Operator<DTYPE>::ResetResult()
    @ref int Operator<DTYPE>::SetIsTensorholder(int pIsParameter)
    @ref int Operator<DTYPE>::SetIsTrainable(int pIsTrainable)
    @ref int Operator<DTYPE>::AddGradient(Tensor<DTYPE> *pTensor)
    */
    int Alloc(Tensor<DTYPE>* pTensor, int pTrainable)
    {
#ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::Alloc(Tensor<DTYPE> *, std::string)" << '\n';
#endif // __DEBUG__

        if (pTensor)
        {
            this->SetResult(pTensor);
        }
        else
        {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__,
                   __FILE__, __LINE__);
            return FALSE;
        }

        this->SetIsTensorholder(TRUE);
        this->SetIsTrainable(pTrainable);
        this->AddGradient(new Tensor<DTYPE>(new Shape(pTensor->GetShape())));

        return TRUE;
    }

    /*!
    @brief 파라미터로 뱓은 변수들로 pTensor를 생성하고 Tensorholder를 설정한다.
    @details 파라미터로 받은 변수들로 pTensor를 생성하고 Result로 설정한다.
    @details 생성한 pTensor의 Shape과 같은 Tensor를 생성하여 Gradient로 설정한다.
    @param pTimeSize 생성할 pTensor의 timesize
    @param pBatchSize 생성할 pTensor의 batchsize
    @param pChannelSize 생성할 pTensor의 channelsize
    @param pRowSize 생성할 pTensor의 rowsize
    @param pColSize 생성할 pTensor의 colsize
    @param pTrainable Training이 가능한지 아닌지 알리는 변수
    @return 성공 시 TRUE.
    @ref int Operator<DTYPE>::ResetResult()
    @ref int Operator<DTYPE>::SetIsTensorholder(int pIsParameter)
    @ref int Operator<DTYPE>::SetIsTrainable(int pIsTrainable)
    @ref Shape *Tensor<DTYPE>::GetShape()
    @ref int Operator<DTYPE>::AddGradient(Tensor<DTYPE> *pTensor)
    */
    int Alloc(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize,
              int pTrainable)
    {
#ifdef __DEBUG__
        std::cout << "Placeholder<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
#endif // __DEBUG__

        Tensor<DTYPE>* pTensor =
            Tensor<float>::Zeros(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        if (pTensor)
        {
            this->SetResult(pTensor);
        }
        else
        {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__,
                   __FILE__, __LINE__);
            return FALSE;
        }

        this->SetIsTensorholder(TRUE);
        this->SetIsTrainable(pTrainable);
        Shape* shapeOfDelta = new Shape(pTensor->GetShape());
        this->AddGradient(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    Tensor<DTYPE>* GetTensor() { return this->GetResult(); }

    /*!
    @brief 파라미터로 받은 Tensor를 Result로 설정한다.
    @param pTensor Result로 설정 할 Tensor.
    */
    void SetTensor(Tensor<DTYPE>* pTensor) { this->SetResult(pTensor); }

    /*!
    @brief 파라미터로 받은 Tensor를 Result로 설정한다.
    @param pTensor Result로 설정 할 Tensor.
    */
    void FeedTensor(Tensor<DTYPE>* pTensor) { this->SetResult(pTensor); }
};

#endif // TENSORHOLDER_H_
