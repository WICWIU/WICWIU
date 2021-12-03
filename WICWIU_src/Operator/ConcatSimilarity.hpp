#ifndef CONCATSIMILARITY_H_
#define CONCATSIMILARITY_H_    value

#include "../Operator.hpp"
#include <cstdio>

/*!
@class ConcatSimilarity ConcatSimilarity class
*/
template<typename DTYPE> class ConcatSimilarity : public Operator<DTYPE>{
private:

    Operator<DTYPE> *m_aKeyMatMul;
    Operator<DTYPE> *m_aKeyMatMulTranspose;
    Operator<DTYPE> *m_aTemp;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *m_aMatMul;

public:

    /**
     * @brief ConcatSimilarity의 생성자
     * @details Bahdanau에 나오는 Alignment model, Loung attention paper에서 Concat Similarity라는 명칭을 사용
     * @details a(si−1, hj ) = v^T * tanh (WaSi−1 + UaHj)
     * @param pKey Bahdanau attention의 key  Operator (모든 Encoder의 hidden 값)
     * @param pWeightV alignment model의 Weight V
     * @param pWeightW alignment model의 Weight W
     * @param pWeightU alignment model의 Weight U
     * @param pQuery Bahdauanu attention의 Query Operator, (t 시점의 decoder hidden)
     * @param pName 사용자가 부여한 Operator이름.
     */
    ConcatSimilarity(Operator<DTYPE> *pKey, Operator<DTYPE> *pWeightV, Operator<DTYPE> *pWeightW, Operator<DTYPE> *pWeightU, Operator<DTYPE> *pQuery, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(5, pKey, pWeightV, pWeightW, pWeightU, pQuery, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "ConcatSimilarity::ConcatSimilarity(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pKey, pWeightV, pWeightW, pWeightU, pQuery);
    }

    /**
     * @brief ConcatSimilarity의 소멸자
     * @details ConcatSimilarity에서 할당했던 값들을 해제한다.
     */
    virtual ~ConcatSimilarity() {
        #ifdef __DEBUG__
        std::cout << "ConcatSimilarity::~ConcatSimilarity()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }
    /**
     * @brief 파라미터로 받은 pKey, pWeightV, pWeightW, pWeightU, pQuery으로 맴버 변수들을 초기화 한다.
     * @details 입력받은 파라미터를 사용하여 연산에 필요한 Operator와 Tensor를 정의한다.
     * @param pKey Bahdanau attention의 key  Operator (모든 Encoder의 hidden 값)
     * @param pWeightV alignment model의 Weight V
     * @param pWeightW alignment model의 Weight W
     * @param pWeightU alignment model의 Weight U
     * @param pQuery Bahdauanu attention의 Query Operator, (t 시점의 decoder hidden) 
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pWeightV, Operator<DTYPE> *pWeightW, Operator<DTYPE> *pWeightU,  Operator<DTYPE> *pQuery) {
        #ifdef __DEBUG__
        std::cout << "ConcatSimilarity::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pQuery->GetResult()->GetTimeSize();
        int batchsize   = pQuery->GetResult()->GetBatchSize();
        int channelsize = pQuery->GetResult()->GetChannelSize();
        int rowsize     = pQuery->GetResult()->GetRowSize();
        int colsize     = pQuery->GetResult()->GetColSize();
        int EncTimeSize = pKey->GetResult()->GetTimeSize();

        m_aKeyMatMul    = new MatMul<DTYPE>(pWeightU, pKey, "ConcatSimilarity_first");      //encoder hidden
        m_aKeyMatMulTranspose = new TransposeTimeWise<DTYPE>(m_aKeyMatMul, 0, 3, "ConcatSimilarity_transpose");
        m_aTemp           = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, colsize), "tempHidden");   //W_a * S_i-1 
        m_aPrevActivate   = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, EncTimeSize, colsize), "_addall");   
        ApplyActivation  = new Tanh<DTYPE>(m_aPrevActivate, "ConcatSimilarity_tanh");
        m_aMatMul       = new BahdanauBroadMatMul<DTYPE>(ApplyActivation, pWeightV, "ConcatSimilarity_last");      

        //For AnalyzeGraph
        pKey->GetOutputContainer()->Pop(m_aKeyMatMul);
        pWeightU->GetOutputContainer()->Pop(m_aKeyMatMul);
        pWeightV->GetOutputContainer()->Pop(m_aMatMul);

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, EncTimeSize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, EncTimeSize));

        return TRUE;
    }


#ifdef __CUDNN__
        /**
         * @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
         * @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
         * @param idOfDevice 사용할 GPU의 id
         */
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

          m_aKeyMatMul->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aKeyMatMulTranspose->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aTemp->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aPrevActivate->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          ApplyActivation->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aMatMul->SetDeviceGPU(this->GetCudnnHandle(), this->GetCublasHandle(), idOfDevice);
      }

#endif  // if __CUDNN__


    void Delete() {
    }

    /**
     * @brief WICWICU는 정적 그래프 구조를 갖기 때문에 추후 실제 Query에 해당하는 Operator를 연결해준다.
     * @param pQuery Query에 해당하는 Operator
     * @return int 
     */
    virtual int SetQuery(Operator<DTYPE>* pQuery){
        std::cout<<"ConcatSimilarity SetQuery"<<'\n';
        std::cout<<this->GetInputContainer()->GetLast()->GetName()<<'\n';
        this->GetInputContainer()->Pop(this->GetInputContainer()->GetLast());
        this->GetInputContainer()->Push(pQuery);

        return TRUE;
    }

    /**
     * @brief ConcatSimilarity의 ForwardPropagate 메소드
     * @details Key와 query의 유사도를 계산한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int ForwardPropagate(int pTime = 0) {

        Tensor<DTYPE> *key = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *query  = this->GetInput()[4]->GetResult();

        Shape *keyShape    = key->GetShape();
        Shape *queryShape  = query->GetShape();

        int keytimesize = key->GetTimeSize();

        if(pTime == 0){
            for(int ti = 0; ti < keytimesize; ti++)    m_aKeyMatMul->ForwardPropagate(ti);

            m_aKeyMatMulTranspose->ForwardPropagate();

            return TRUE;        
        }

        Tensor<DTYPE> *weight  = this->GetInput()[2]->GetResult();      
        Tensor<DTYPE> *temp    = m_aTemp->GetResult();

        Shape *weightShape = weight->GetShape();
        Shape *tempShape   = temp->GetShape();

        int batchsize = query->GetBatchSize();
        int rowsize   = query->GetColSize();        
        int colsize   = query->GetColSize();

        for(int ba = 0; ba < batchsize; ba++) {
            for(int ro = 0; ro < rowsize; ro++){
                for(int co = 0; co < colsize; co++) {
                    (*temp)[Index5D(tempShape, pTime, ba, 0, 0, ro)] +=
                        (*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)] * (*weight)[Index5D(weightShape, 0, 0, 0, ro, co)];
                }
            }
        }

        Tensor<DTYPE> *UH = m_aKeyMatMulTranspose->GetResult();      
        Shape *UHShape = UH->GetShape();

        Tensor<DTYPE> *addResult = m_aPrevActivate->GetResult();      
        Shape *addResultShape = addResult->GetShape();

        colsize = temp->GetColSize();

        for(int ti=0; ti< keytimesize; ti++){
            for(int ba = 0; ba < batchsize; ba++) {
                    for(int co = 0; co < colsize; co++) {
                        (*addResult)[Index5D(addResultShape, pTime, ba, 0, ti, co)] +=
                            (*temp)[Index5D(tempShape, pTime, ba, 0, 0, co)] * (*UH)[Index5D(UHShape, 0, ba, 0, ti, co)];            
                    }
            }
        }

        ApplyActivation->ForwardPropagate(pTime);        


        m_aMatMul->ForwardPropagate(pTime);             


        Tensor<DTYPE> *_result = m_aMatMul->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();


        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();
        Shape *_ResultShape = _result->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int co = 0; co < colSize; co++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, co)] = (*_result)[Index5D(_ResultShape, pTime, ba, 0, co, 0)];
            }
        }

        return TRUE;
    }

    /**
     * @brief ConcatSimilarity의 BackPropagate 메소드
     * @details alignment model의 미분값을 계산하여 key_gradient와 query_gradient에 각각 더해 넣는다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int BackPropagate(int pTime = 0) {

      Tensor<DTYPE> *key = this->GetInput()[0]->GetGradient();
      Tensor<DTYPE> *query  = this->GetInput()[4]->GetGradient();

      Shape *queryShape  = query->GetShape();


      int keytimesize = key->GetTimeSize();
      int batchsize = query->GetBatchSize();

      Tensor<DTYPE> *_result = m_aMatMul->GetGradient();
      Tensor<DTYPE> *result  = this->GetGradient();


      int colSize        = result->GetColSize();
      Shape *ResultShape = result->GetShape();
      Shape *_ResultShape = _result->GetShape();

      for(int ba=0; ba<batchsize; ba++){
          for (int co = 0; co < colSize; co++) {
              (*_result)[Index5D(_ResultShape, pTime, ba, 0, co, 0)] = (*result)[Index5D(ResultShape, pTime, ba, 0, 0, co)];
          }
      }

      m_aMatMul->BackPropagate(pTime);
      ApplyActivation->BackPropagate(pTime);


      Tensor<DTYPE> *addResult = m_aPrevActivate->GetGradient();      
      Shape *addResultShape = addResult->GetShape();


      Tensor<DTYPE> *UH = m_aKeyMatMulTranspose->GetGradient();      
      Shape *UHShape = UH->GetShape();

      Tensor<DTYPE> *temp    = m_aTemp->GetGradient();
      Shape *tempShape   = temp->GetShape();

      int colsize = temp->GetColSize();


      for(int ti=0; ti< keytimesize; ti++){
          for(int ba = 0; ba < batchsize; ba++) {
                  for(int co = 0; co < colsize; co++) {

                      (*UH)[Index5D(UHShape, 0, ba, 0, ti, co)] +=
                          (*addResult)[Index5D(addResultShape, pTime, ba, 0, ti, co)] ;

                      (*temp)[Index5D(tempShape, pTime, ba, 0, 0, co)] +=
                          (*addResult)[Index5D(addResultShape, pTime, ba, 0, ti, co)] ;
                  }
          }
      }


      if(pTime == 0){
          m_aKeyMatMulTranspose->BackPropagate();

          for(int ti = keytimesize-1; ti > 0; ti--)    m_aKeyMatMul->BackPropagate(ti);
          return TRUE;
      }

      Tensor<DTYPE> *weight  = this->GetInput()[2]->GetResult();
      Tensor<DTYPE> *weightGradient  = this->GetInput()[2]->GetGradient();

      query  = this->GetInput()[4]->GetResult();
      Tensor<DTYPE> *queryGradient  = this->GetInput()[4]->GetGradient();

      Shape *weightShape = weight->GetShape();


      batchsize = query->GetBatchSize();
      int rowsize   = query->GetColSize();
      colsize   = query->GetColSize();


        for(int ba = 0; ba < batchsize; ba++) {
            for(int ro = 0; ro < rowsize; ro++){
                for(int co = 0; co < colsize; co++) {

                    (*weightGradient)[Index5D(weightShape, 0, 0, 0, ro, co)] +=
                       (*temp)[Index5D(tempShape, pTime, ba, 0, 0, ro)]  * (*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)];

                    (*queryGradient)[Index5D(queryShape, pTime-1, ba, 0, 0, co)] +=
                        (*temp)[Index5D(tempShape, pTime, ba, 0, 0, ro)] * (*weight)[Index5D(weightShape, 0, 0, 0, ro, co)];
                }
            }
        }


        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__

    /**
     * @brief 맴버변수로 선언한 Operator들의 Result값을 초기화한다.
     * @return int 
     */
    int ResetResult() {

      m_aKeyMatMul->ResetResult();
      m_aKeyMatMulTranspose->ResetResult();
      m_aTemp->ResetResult();
      m_aPrevActivate->ResetResult();
      ApplyActivation->ResetResult();
      m_aMatMul->ResetResult();

      Tensor<DTYPE> *result = this->GetResult();
      result->Reset();

      return TRUE;

    }
    /**
     * @brief 맴버변수로 선언한 Operator들의 Gradient값을 초기화한다.
     * @return int 
     */
    int ResetGradient() {

      m_aKeyMatMul->ResetGradient();
      m_aKeyMatMulTranspose->ResetGradient();
      m_aTemp->ResetGradient();
      m_aPrevActivate->ResetGradient();
      ApplyActivation->ResetGradient();
      m_aMatMul->ResetGradient();

      Tensor<DTYPE> *grad = this->GetGradient();
      grad->Reset();

       return TRUE;

    }

};

#endif  // DOTSIMILARITY_H_
