#ifndef TRIPLETLOSS_H_
#define TRIPLETLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class TripletLoss : public LossFunction<DTYPE>{
private:
    DTYPE m_margin;
    DTYPE **m_LossPerSample;

    int m_NumOfAnchorSample;
    int m_timesize;


public:
    TripletLoss(Operator<DTYPE> *pOperator, DTYPE margin, int stepsPerUpdate, std::string pName = "NO NAME")
     : LossFunction<DTYPE>(pOperator, NULL, pName) {
        #ifdef __DEBUG__
        std::cout << "TripletLoss::TripletLoss(LossFunction<DTYPE> * 3, float, )" << '\n';
        #endif  // __DEBUG__
        this->stepsPerUpdate_ = stepsPerUpdate;

        Alloc(pOperator, margin);
    }

    TripletLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE margin, std::string pName = "NO NAME")
     : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "TripletLoss::TripletLoss(LossFunction<DTYPE> * 3, float, )" << '\n';
        #endif  // __DEBUG__

        Alloc(pOperator, margin);
    }

    ~TripletLoss() {
        #ifdef __DEBUG__
        std::cout << "TripletLoss::~TripletLoss()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc(Operator<DTYPE> *pOperator, DTYPE margin) {
        #ifdef __DEBUG__
        std::cout << "TripletLoss::Alloc( Operator<DTYPE> *, float)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_margin = margin;
        m_timesize = timesize;

        m_LossPerSample = new DTYPE *[timesize];

        m_NumOfAnchorSample = (batchsize / 3);

        for (int i = 0; i < timesize; i++) {
            m_LossPerSample[i] = new DTYPE[m_NumOfAnchorSample/* * channelsize * rowsize * colsize*/];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, m_NumOfAnchorSample, 1, 1, 1));

        return TRUE;
    }

    virtual void Delete() {
      if (m_LossPerSample) {
          delete m_LossPerSample;
          m_LossPerSample = NULL;
      }
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> &input         = *this->GetTensor();
        Tensor<DTYPE> &result        = *this->GetResult();
        Tensor<DTYPE> &label         = *this->GetLabel()->GetResult();

        int batchsize   = input.GetBatchSize();
        int channelsize = input.GetChannelSize();
        int rowsize     = input.GetRowSize();
        int colsize     = input.GetColSize();
        int featureDim = channelsize * rowsize * colsize;

        int ti = pTime;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < m_NumOfAnchorSample; ba++) {
            float dis_pos = 0.f;
            float dis_neg = 0.f;

            int anc_idx = (pTime * batchsize + ba * 3) * featureDim;
            int anc_limit = anc_idx + featureDim;

            int pos_idx = anc_idx + featureDim;
            int neg_idx = pos_idx + featureDim;

            for (; anc_idx < anc_limit; anc_idx++, pos_idx++, neg_idx++) {
                float d1 = (input[anc_idx] - input[pos_idx]);
                dis_pos += d1 * d1;

                float d2 = (input[anc_idx] - input[neg_idx]);
                dis_neg += d2 * d2;
            }

            float loss = (dis_pos - dis_neg) + m_margin;
            // loss /= featureDim;

            if (loss < 0.f)
                loss = 0.f;

            if (loss > 0.f) {
                result[ti * m_NumOfAnchorSample + ba] = loss; 
                m_LossPerSample[ti][ba] = 1;
            } else {
                result[ti * m_NumOfAnchorSample + ba] = 0;
                m_LossPerSample[ti][ba] = 0;
            }
        }

        return &result;
    }

    Tensor<DTYPE>* BackPropagate(int pTime = 0) {

        Tensor<DTYPE> &input          = *this->GetTensor();
        Tensor<DTYPE> &input_gradient = *this->GetOperator()->GetGradient();
        Tensor<DTYPE> &result         = *this->GetResult();

        int batchsize   = input_gradient.GetBatchSize();
        int channelsize = input_gradient.GetChannelSize();
        int rowsize     = input_gradient.GetRowSize();
        int colsize     = input_gradient.GetColSize();

        int featureDim = channelsize * rowsize * colsize;

        for (int ba = 0; ba < m_NumOfAnchorSample; ba++) {
            int anc_idx = (pTime * batchsize + ba * 3) * featureDim;
            int anc_limit = anc_idx + featureDim;
            int pos_idx = anc_idx + featureDim;
            int neg_idx = pos_idx + featureDim;

            if (m_LossPerSample[pTime][ba]) {
                for (; anc_idx < anc_limit; anc_idx++, pos_idx++, neg_idx++) {
                    input_gradient[anc_idx] = (2.f * (input[neg_idx] - input[pos_idx])) / featureDim;
                    input_gradient[pos_idx] = (2.f * (input[pos_idx] - input[anc_idx])) / featureDim;
                    input_gradient[neg_idx] = (2.f * (input[anc_idx] - input[neg_idx])) / featureDim;
                }
            } else {
                memset(&input_gradient[anc_idx], 0, featureDim * sizeof(DTYPE) * 3);
            }
        }

        return NULL;
    }

   #ifdef __CUDNN__

       Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
           this->ForwardPropagate();
           return NULL;
       }


       Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
           this->BackPropagate();
           return NULL;
       }

   #endif  // __CUDNN__

};
#endif  // TRIPLETLOSS_H_