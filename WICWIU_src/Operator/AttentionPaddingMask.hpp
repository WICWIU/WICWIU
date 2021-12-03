#ifndef ATTENTION_PADDING_MASK_HPP__
#define ATTENTION_PADDING_MASK_HPP__

#include "../Operator.hpp"

/**
 * @brief Input Tensor에 적합한 AttentionPaddingMask를 생성하는 Operator Class.
 */
template <typename DTYPE> class AttentionPaddingMask : public Operator<DTYPE> {
private:
    Tensor<DTYPE> *m_aSubsequentMask;
    ///< Decoder의 Attention연산에 사용되는 미래 정보를 가리는 Mask
    DTYPE          m_paddingTok;
    ///< Padding Token
    int            m_IsDecoder;
    ///< 해당 클래스가 Decoder를 위한 인스턴스인지에 대한 플래그

public:
    /**
     * @brief AttentionPaddingMask의 생성자
     * @details 파라미터로 받은 pInput, vocabLength, mask, IsDecoder를 Alloc한다.
     * @param pInput Alloc할 대상 Operator
     * @param vocabLength Vocabulary의 전체 Unique 단어 개수
     * @param mask Mask를 의미하는 값
     * @param IsDecoder 해당 인스턴스가 Decoder를 위한 것인지 나타내는 플래그.
     */
    AttentionPaddingMask(Operator<DTYPE> *pInput, int vocabLength, DTYPE mask = 0.F, int IsDecoder = FALSE, std::string Name = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, Name, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "AttentionPaddingMask<DTYPE>::AttentionPaddingMask(Operator<DTYPE> , int , int , int , std::string , int )" << '\n';
#endif // __DEBUG__

        m_aSubsequentMask = NULL;
        m_paddingTok      = 0;
        m_IsDecoder       = FALSE;

        Alloc(pInput, vocabLength, mask, IsDecoder);
    }

    /**
     * @brief 파라미터로 받은 pInput, vocabLength, mask, IsDecoder로부터 멤버 변수들을 초기화한다.
     * @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
     * @details SubsequentMask의 경우, Time Size * Time Size 2D Tensor이다.
     * @param pInput 생성할 Tensor의 Shape정보를 가진 Operator
     * @param vocabLength Vocabulary의 전체 Unique 단어 개수
     * @param mask Mask를 의미하는 값
     * @param IsDecoder 해당 인스턴스가 Decoder를 위한 것인지 나타내는 플래그.
     * @return 성공 시 TRUE 
     */
    int Alloc(Operator<DTYPE> *pInput, int vocabLength, int mask, int IsDecoder) {
        #ifdef __DEBUG__
            std::cout << "AttentionPaddingMask<DTYPE>::Alloc(Operator<DTYPE> *, int " << '\n';
        #endif // __DEBUG__

        m_paddingTok = mask;
        m_IsDecoder  = IsDecoder;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int colsize     = pInput->GetResult()->GetColSize();
        int rowsize     = colsize;

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        m_aSubsequentMask = Tensor<DTYPE>::Constants(1, 1, 1, colsize, colsize, 1);

        if (IsDecoder)
            m_aSubsequentMask->TriangleLower(0, 0.F);

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_aSubsequentMask->SetDeviceGPU(this->GetDeviceID());
    }
#endif
    /**
     * @brief AttentionPaddingMask의 ForwardPropagate 메소드.
     * @details input Tensor를 바탕으로 Attention 연산에서 사용되는 Padding + Subsequent Mask를 생성
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *inputTenShape          = input->GetShape();
        Shape *resultTenShape         = result->GetShape();
        Shape *SubsequentMaskTenShape = m_aSubsequentMask->GetShape();

        int ti = pTime;

        DTYPE fill = 1.F;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int co = 0; co < colsize; co++) {
                    if ((*input)[Index5D(inputTenShape, ti, ba, 0, 0, co)] == m_paddingTok)
                        fill = 0.F;
                    else
                        fill = 1.F;

                    for (int ro = 0; ro < rowsize; ro++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] =
                          1 - fill * (*m_aSubsequentMask)[Index5D(SubsequentMaskTenShape, 0, 0, 0, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

    /**
     * @brief AttentionPaddingMask의 BackPropagate 메소드.
     * @details 딱히 하는 일은 없다.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);
#endif // __CUDNN__
};


/*!
@class AttentionPaddingMaskRNN AttentionPaddingMaskRNN class
*/
template<typename DTYPE> class AttentionPaddingMaskRNN : public Operator<DTYPE> {
private:
  DTYPE m_paddingTok;
public:
/**
 * @brief AttentionPaddingMaskRNN의 생성자
 * @details 파라미터로 받은 pInput, make, name으로 Alloc한다.
 * @param pInput Mask를 적용할 Operator
 * @param mask Mask의 위치를 갖고있는 Opeator
 * @param Name 사용자가 부여한 Operator 이름
 */
  AttentionPaddingMaskRNN(Operator<DTYPE> *pInput, DTYPE mask = 0.F, std::string Name = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, Name, pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "AttentionPaddingMaskRNN<DTYPE>::AttentionPaddingMaskRNN(Operator<DTYPE> , int , int , int , std::string , int )" << '\n';
    #endif  // __DEBUG__

    m_paddingTok = 0;

    Alloc(pInput, mask);
  }

 /**
 * @brief 파라미터로 받은 pInput, make, name으로부터 맴버 변수들을 초기화 한다.
 * @details esult와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
 * @param pInput Mask를 적용할 Operator
 * @param mask Mask의 위치를 갖고있는 Opeator
 * @return int 
 */
  int Alloc(Operator<DTYPE> *pInput, int mask) {
    #ifdef __DEBUG__
    std::cout << "AttentionPaddingMaskRNN<DTYPE>::Alloc(Operator<DTYPE> *, int " << '\n';
    #endif  // __DEBUG__

    m_paddingTok = mask;

    int timesize    = pInput->GetResult()->GetTimeSize();
    int batchsize   = pInput->GetResult()->GetBatchSize();
    int channelsize = pInput->GetResult()->GetChannelSize();
    int rowsize     = pInput->GetResult()->GetRowSize();
    //int colsize     = pInput->GetResult()->GetTimeSize();

    this->SetResult(new Tensor<DTYPE>(1, batchsize, channelsize, rowsize, timesize));
    this->SetDelta(new Tensor<DTYPE>(1, batchsize, channelsize, rowsize, timesize));

  }


  /**
   * @brief AttentionPaddingMaskRNN의 Forward 함수
   * @details 0 : real value, 1 : padding
   * @param pTime 
   * @return int 
   */
  int ForwardPropagate(int pTime = 0) {

    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int inputTimeSize = input->GetTimeSize();
    int inputColSize = input->GetColSize();

    int batchsize     = result->GetBatchSize();
    int channelsize   = result->GetChannelSize();
    int rowsize       = result->GetRowSize();
    int colsize       = result->GetColSize();

    Shape *inputTenShape  = input->GetShape();
    Shape *resultTenShape = result->GetShape();

    for (int ti = 0; ti < inputTimeSize; ti++){
        for (int ba = 0; ba < batchsize; ba++) {
              if((*input)[Index5D(inputTenShape, ti, ba, 0, 0, 0)] == m_paddingTok)
                  (*result)[Index5D(resultTenShape, 0, ba, 0, 0, ti)] = 1;
              else
                  (*result)[Index5D(resultTenShape, 0, ba, 0, 0, ti)] = 0;
        }
    }

    return TRUE;
  }

  int BackPropagate(int pTime = 0) {
    return TRUE;
  }

#ifdef __CUDNN__
  int ForwardPropagateOnGPU(int pTime) {
      this->ForwardPropagate(pTime);
      return TRUE;
  }

  int BackPropagateOnGPU(int pTime = 0) {
      this->BackPropagate(pTime);
      return TRUE;
  }

#endif  // __CUDNN__
};

#endif