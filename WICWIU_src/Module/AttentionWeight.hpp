#ifndef __ATTENTION_WEIGHT_HPP__
#define __ATTENTION_WEIGHT_HPP__

#include "../Module.hpp"

template<typename DTYPE> class DotProductAttentionWeight : public Module<DTYPE> {
private:
    Operator<DTYPE> *m_pAttention;

public:
    DotProductAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate = 0.0f, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "DotProductAttentionWeight<DTYPE>::DotProductAttentionWeight(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int, float, std::string )" << '\n';
#endif  // __DEBUG__
        Alloc(pKey, pQuery, pMask, d_h, droprate, pName);
    }

    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate, std::string pName) {
#ifdef __DEBUG__
        std::cout << "DotProductAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *, bool , std::string )" << '\n';
#endif  // __DEBUG__

        this->SetInput(3, pKey, pQuery, pMask);

        Operator<DTYPE> *out = NULL;

        // #1. MatMul
        out = new BroadMatMul<DTYPE>(pQuery, new Transpose<DTYPE>(pKey, 3, 4, pName + "_KeyTranspose"), pName+"_KeyTQueryMatMul");
        // #2. Scale
        out = new Scale<DTYPE>(out, 1.f/sqrt((DTYPE)d_h), pName + "_Scale");
        // #3. pMask
        if(pMask) {
            out = new MaskedFill<DTYPE>(out, pMask, -1e9, pName + "_MaskedFill");
        }
        // #4. softmax

        out = new Softmax1D<DTYPE>(out, 1e-6f, 4, pName+"_Softmax1D");
        m_pAttention = out;

        out = new Dropout<DTYPE>(out, droprate, pName+"_Dropout");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetAttention() {
        return m_pAttention;
    }
};


//key   : encoder hidden query : decoder hidden
/*!
 * @class Bahdanau Attention weight을 계산하는 모듈을 생성하는 클래스
 * @details Key, Query, Value를 받아 Bahdanau Attention Weight을 계산한다.
*/
template<typename DTYPE> class BahdanauAttentionWeight : public Module<DTYPE> {
private:

public:

    /*!
     * @brief BahdanauAttentionWeight 클래스 생성자
     * @details BahdanauAttentionWeight 클래스의 Alloc 함수를 호출한다.
    */
    BahdanauAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) : Module<DTYPE>(pName) {
        Alloc(pKey, pQuery, pMask, pName);
    }

    virtual ~BahdanauAttentionWeight() {}

    /**
     * @brief BahdanauAttentionWeight 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Key와 Query를 받아 Bahdanau attetnion weight값을 계산한다.
     * @param pKey Key에 해당하는 Operator (encoder hidden)
     * @param pQuery Query에 해당하는 Operator (decoder hidden)
     * @param pMask tention연산 시 연산에 참여하지 않는 위치 정보를 갖고있는 Operator
     * @param pName Module의 이름
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) {
        this->SetInput(3, pKey, pQuery, pMask);    

        int channelSize    = pKey->GetResult()->GetChannelSize();
        int rowSize        = pKey->GetResult()->GetRowSize();
        int EncoderColSize = pKey->GetResult()->GetColSize();
        int DecoderColSize = pQuery->GetResult()->GetColSize();

        //weight
        Tensorholder<DTYPE> *pWeightV = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, DecoderColSize, 1, 0.0, 0.01), "Bahdanau_Weight_V_" + pName);       //BroadMatMul
        Tensorholder<DTYPE> *pWeightW = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, DecoderColSize, DecoderColSize, 0.0, 0.01), "Bahdanau_Weight_W_" + pName);
        Tensorholder<DTYPE> *pWeightU = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, DecoderColSize, EncoderColSize, 0.0, 0.01), "Bahdanau_Weight_U_" + pName);

        Operator<DTYPE> *out = NULL;

        out = new ConcatSimilarity<DTYPE>(pKey, pWeightV, pWeightW, pWeightU, pQuery, pName+"_similarity");

        if(pMask) {
          out = new MaskedFill<DTYPE>(out, pMask, -1e9, pName+"_pMask");
        }

        out = new Softmax<DTYPE>(out, pName+"_attention_weight");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    virtual int SetQuery(Operator<DTYPE> * pQuery){

        std::cout<<"BahdanauAttentionWeight SetQuery"<<'\n';

        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();
        (*ExcutableOperator)[0]->SetQuery(pQuery);

        return true;

    }
};

#endif //__ATTENTION_WEIGHT_HPP__
