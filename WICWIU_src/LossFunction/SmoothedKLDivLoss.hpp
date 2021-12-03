#ifndef S_KLDIVLOSS_H_
#define S_KLDIVLOSS_H_ value

#include "../LossFunction.hpp"

template <typename DTYPE> class SmoothedKLDivLoss : public LossFunction<DTYPE> {
private:
    int   m_VocabSize;
    float m_Smoothing;
    int   m_timeIndex;
    Tensor<DTYPE> *m_aSmoothedLabel;

#ifdef __CUDNN__

    DTYPE *m_pDevInput, *m_pDevLabel, *m_pDevOutput, *m_pDevInputDelta, *m_pDevSmoothedLabel;

#endif

public:
    SmoothedKLDivLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, float smoothing, int vocabSize, int timeIndex = 0,
                      std::string pName = "NoName")
        : LossFunction<DTYPE>(pOperator, pLabel) {
#ifdef __DEBUG__
        std::cout << "SmoothedKLDivLoss::SmoothedKLDivLoss(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
#endif // __DEBUG__
        m_VocabSize = vocabSize;
        m_Smoothing = smoothing;
        m_timeIndex = timeIndex;
        this->Alloc(pOperator, pLabel);
    }

    ~SmoothedKLDivLoss() {
#ifdef __DEBUG__
        std::cout << "SmoothedKLDivLoss::~SmoothedKLDivLoss()" << '\n';
#endif // __DEBUG__
        Delete();
    }

    virtual int Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) {
#ifdef __DEBUG__
        std::cout << "SmoothedKLDivLoss::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
#endif // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        if (m_timeIndex == 0)
            this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));
        else if (m_timeIndex == 2)
            this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, 1, 1));
        m_aSmoothedLabel = Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize);

        return TRUE;
    }
#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_aSmoothedLabel->SetDeviceGPU(this->GetDeviceID());
    }
#endif

    virtual void Delete() {
        if (m_aSmoothedLabel) {
            delete m_aSmoothedLabel;
        }
        m_aSmoothedLabel = NULL;
    }

    Tensor<DTYPE> *ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        Shape *shape = input->GetShape();
        Shape *resultShape = result->GetShape();

        int ti = pTime;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int index                = Index5D(shape, ti, ba, ch, ro, co);
                        (*m_aSmoothedLabel)[index] = (1 - m_Smoothing) * (*label)[index] + m_Smoothing / m_VocabSize;
                    }
                }
            }
        }

        if (m_timeIndex == 0) {
            ti = pTime;

            for (int ba = 0; ba < batchsize; ba++) {
                int lossIndex = Index5D(resultShape, ti, ba, 0, 0, 0);

                DTYPE batchSum = 0;
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int index = Index5D(shape, ti, ba, ch, ro, co);
                            batchSum += (*m_aSmoothedLabel)[index] * (std::log((*m_aSmoothedLabel)[index]) - std::log((*input)[index]));
                        }
                    }
                }
                (*result)[lossIndex] = batchSum;
            }
        }
        else if (m_timeIndex == 2) {
            ti = pTime;
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    int lossIndex = Index5D(resultShape, ti, ba, ch, 0, 0);

                    DTYPE batchSum = 0;
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            int index = Index5D(shape, ti, ba, ch, ro, co);
                            if ((*m_aSmoothedLabel)[index] <= 0) {
                                std::cout << "label (" << (*m_aSmoothedLabel)[index] << ") is lower than 0" << std::endl;
                            }
                            if ((*input)[index] <= 0) {
                                std::cout << "input (" << (*input)[index] << ") is lower than 0" << std::endl;
                            }
                            batchSum += (*m_aSmoothedLabel)[index] * (std::log((*m_aSmoothedLabel)[index]) - std::log((*input)[index]));
                        }
                    }
                    (*result)[lossIndex] = batchSum;
                }
            }
        }
        else {
            std::cout << "Unsupported Time Index!" << '\n';
        }
        // std::cout << result << '\n';
        return result;
    }

    Tensor<DTYPE> *BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        Shape *shape = input->GetShape();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        
        int ti = pTime;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int index = Index5D(shape, ti, ba, ch, ro, co);
                        // (*input_delta)[index] = (1-m_Smoothing)*(*temp_delta)[index];
                        (*input_delta)[index] += -((1.f - m_Smoothing) * (*label)[index] + m_Smoothing / m_VocabSize) / (*input)[index];
                    }
                }
            }
        }

        return NULL;
    }

#ifdef __CUDNN__
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime);

    Tensor<DTYPE>* BackPropagateOnGPU(int pTime);
#endif
};

#endif // S_KLDIVLOSS_H_
