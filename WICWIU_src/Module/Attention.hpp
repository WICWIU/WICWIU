#ifndef __ATTENTION_HPP__
#define __ATTENTION_HPP__

#include "../Module.hpp"


template<typename DTYPE> class MultiHeadAttention : public Module<DTYPE> {
private:
    int d_k;
    Operator<DTYPE> *m_pAttentionWeight;

public:
    MultiHeadAttention(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask, float droprate, int d_model, int n_head, std::string pName = "NO NAME") {
#ifdef __DEBUG__
        std::cout << "MultiHeadAttention<DTYPE>::MultiHeadAttention(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, float, int , int , std::string )" << '\n';
#endif  // __DEBUG__
        Alloc(pKey, pQuery, pValue, pMask, droprate, d_model, n_head, pName);
    }

    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask, float droprate, int d_model, int n_head, std::string pName) {
#ifdef __DEBUG__
        std::cout << "MultiHeadAttention<DTYPE>::MultiHeadAttention(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, float, int , int , std::string )" << '\n';
#endif  // __DEBUG__

        this->SetInput(4, pKey, pQuery, pValue, pMask);
        Operator<DTYPE> *out = NULL;
        Shape *pInputShape   = pKey->GetResult()->GetShape();

        int timesize    = (*pInputShape)[0];
        int batchsize   = (*pInputShape)[1];
        int channelsize = (*pInputShape)[2];

        d_k = d_model/n_head;

        // #1 : Linear
        pKey = new Linear<DTYPE>(pKey, d_model, d_model, FALSE, pName+"_Key_Linear");
        pQuery = new Linear<DTYPE>(pQuery, d_model, d_model, FALSE, pName+"_Query_Linear");
        pValue = new Linear<DTYPE>(pValue, d_model, d_model, FALSE, pName+"_Value_Linear");

        pKey = new ReShape<DTYPE>(pKey, timesize, batchsize, channelsize, n_head, d_k, pName+"_ReShape_Key");
        pQuery = new ReShape<DTYPE>(pQuery, timesize, batchsize, channelsize, n_head, d_k, pName+"_ReShape_Query");
        pValue = new ReShape<DTYPE>(pValue, timesize, batchsize, channelsize, n_head, d_k, pName+"_ReShape_Value");

        pKey   = new Transpose<DTYPE>(pKey, 2, 3, pName+"_Key_Transpose");
        pQuery = new Transpose<DTYPE>(pQuery, 2, 3, pName+"_Query_Transpose");
        pValue = new Transpose<DTYPE>(pValue, 2, 3, pName+"_Value_Transpose");

        // #2 : Attention
        out = new DotProductAttentionWeight<DTYPE>(pKey, pQuery, pMask, n_head, droprate, pName+"_AttentionWeight");
        m_pAttentionWeight = out;
        out = new BroadMatMul<DTYPE>(out, pValue, pName+"_AttentionWeight*Value");

        // #3 : Concat & Linear
        out = new ReShape<DTYPE>(new Transpose<DTYPE>(out, 2, 3, pName+"attentionTranspose"), timesize, batchsize, channelsize, 1, d_model, pName+"_Concatenate");
        out = new Linear<DTYPE>(out, d_model, d_model, FALSE, pName+"_MultiheadAttentionLinaer");
        
        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetAttentionWeight() {
        return m_pAttentionWeight;
    }


};


/*!
 * @class Bahdanau Attention의 기능을 수행하는 모듈을 생성하는 클래스
 * @details Key, Query, Value를 받아 context vector를 계산한다.
*/
template<typename DTYPE> class BahdanauAttention : public Module<DTYPE>{
private:

public:

    /*!
     * @brief BahdanauAttention 클래스 생성자
     * @details BahdanauAttention 클래스의 Alloc 함수를 호출한다.
    */
    BahdanauAttention(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pKey, pQuery, pValue, pMask, pName);
    }

    virtual ~BahdanauAttention() {}

    /**
     * @brief Bahdanau Attention 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Attention Weight값과 Value값을 사용하여 context vector 계산을 수행한다.
     * @param pKey Key에 해당하는 Operator (Encoder hidden)
     * @param pQuery Query에 해당하는 Operator (Decoder hidden)
     * @param pValue Value에 해당하는 Operator (Encoder hidden)
     * @param pMask Attention연산 시 연산에 참여하지 않는 위치 정보를 갖고있는 Operator
     * @param pName Module의 이름
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask,  std::string pName) {

        this->SetInput(4, pKey, pQuery, pValue, pMask);

        //Bahdanau attetnion
        Operator<DTYPE> *out = new BahdanauAttentionWeight<DTYPE>(pKey, pQuery, pMask, pName+"_Bahdanau_AttentionWeight");  

        Operator<DTYPE> *transpose = new TransposeTimeWise<DTYPE>(pValue, 0 ,3, "BahdanauAttention_transpose");       

        out = new BahdanauBroadMatMul<DTYPE>(out, transpose, "context vector");      

        this->AnalyzeGraph(out);

        return TRUE;
    }

    virtual int SetQuery(Operator<DTYPE> * pQuery){

        std::cout<<"BahdanauAttention SetQuery"<<'\n';

        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        (*ExcutableOperator)[1]->SetQuery(pQuery);

    }

};


#endif //__ATTENTION_HPP__
