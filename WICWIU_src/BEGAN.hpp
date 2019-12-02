#ifndef BEGAN_H_
#define BEGAN_H_

#include "GAN.hpp"

template<typename DTYPE> class BEGAN : public GAN<DTYPE> {
private:
    float m_LossX;
    float m_LossG;

    float m_k;
    float m_lamda;
    float m_gamma;

    float m_ConvergenceMeasure;

public:
    BEGAN();
    virtual ~BEGAN();

    int                                 SetBEGANParameter(float pK, float pLamda, float pGamma);

    virtual int                         TrainGeneratorOnCPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutRealOnCPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutFakeOnCPU();

    virtual int                         TrainGeneratorOnGPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutRealOnGPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutFakeOnGPU();

    float                               SaveLossX();
    float                               SaveLossG();

    int                                 MultiplyKOnOutput();
    int                                 MultiplyKOnGradient();

    float                               UpdateK();

    float                               ComputeConvergenceMeasure();
};

template<typename DTYPE> BEGAN<DTYPE>::BEGAN() : GAN<DTYPE>() {
    #ifdef __DEBUG__
    std::cout << "BEGAN<DTYPE>::BEGAN()" << '\n';
    #endif  // __DEBUG__

    float m_LossX = 0.f;
    float m_LossG = 0.f;

    float m_k = 0.f;
    float m_lamda = 0.f;
    float m_gamma = 0.f;

    float m_ConvergenceMeasure = 0.f;
}

template<typename DTYPE> BEGAN<DTYPE>::~BEGAN(){
    #ifdef __DEBUG__
    std::cout << "BEGAN<DTYPE>::~BEGAN()" << '\n';
    #endif  // __DEBUG__
}

template<typename DTYPE> int BEGAN<DTYPE>::SetBEGANParameter(float pK, float pLamda, float pGamma){
    m_k = pK;
    m_lamda = pLamda;
    m_gamma = pGamma;

    return TRUE;
}

template<typename DTYPE> int BEGAN<DTYPE>::TrainGeneratorOnCPU(){
    this->ResetResult();
    this->ResetGradient();
    this->ResetGeneratorLossFunctionResult();
    this->ResetGeneratorLossFunctionGradient();

    this->GetSwitch()->SetSwitchNumber(FAKE);
    this->ForwardPropagate();
    // std::cout << ", G : D(G(z)) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)] << '\n';
    this->GetGeneratorLossFunction()->ForwardPropagate();
    this->SaveLossG();
    this->GetGeneratorLossFunction()->BackPropagate();
    this->BackPropagate();

    this->GetGeneratorOptimizer()->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int BEGAN<DTYPE>::ComputeGradientOfDiscriminatorAboutRealOnCPU(){
    this->ResetResult();
    this->GetDiscriminator()->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REAL);
    this->GetSwitch()->SetSwitchNumber(REAL);
    this->GetSwitch()->ForwardPropagate();
    this->GetDiscriminator()->ForwardPropagate();
    // std::cout << "D: D(x) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
    this->GetDiscriminatorLossFunction()->ForwardPropagate();
    this->SaveLossX();
    this->GetDiscriminatorLossFunction()->BackPropagate();
    this->GetDiscriminator()->BackPropagate();
}

template<typename DTYPE> int BEGAN<DTYPE>::ComputeGradientOfDiscriminatorAboutFakeOnCPU(){
    this->ResetResult();
    this->GetDiscriminator()->ResetGradient();
    // this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(FAKE);
    this->GetSwitch()->SetSwitchNumber(FAKE);
    this->ForwardPropagate();
    // std::cout << ", D : D(G(z)) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

    // this->MultiplyKOnOutput();
    this->GetDiscriminatorLossFunction()->ForwardPropagate();
    this->GetDiscriminatorLossFunction()->BackPropagate();
    this->MultiplyKOnGradient();

    this->GetDiscriminator()->BackPropagate();
}

template<typename DTYPE> int BEGAN<DTYPE>::TrainGeneratorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ResetGradient();
        this->ResetGeneratorLossFunctionResult();
        this->ResetGeneratorLossFunctionGradient();

        this->GetSwitch()->SetSwitchNumber(FAKE);
        this->ForwardPropagateOnGPU();
        // std::cout << ", G : D(G(z)) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)] << '\n';
        this->GetGeneratorLossFunction()->ForwardPropagateOnGPU();
        this->SaveLossG();
        this->GetGeneratorLossFunction()->BackPropagateOnGPU();
        this->BackPropagateOnGPU();

        this->GetGeneratorOptimizer()->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int BEGAN<DTYPE>::ComputeGradientOfDiscriminatorAboutRealOnGPU(){
    this->ResetResult();
    this->ResetGradient();
    // this->GetDiscriminator()->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REAL);
    this->GetSwitch()->SetSwitchNumber(REAL);
    this->GetSwitch()->ForwardPropagateOnGPU();
    this->GetDiscriminator()->ForwardPropagateOnGPU();
    // std::cout << "\nD: D(x) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

    this->GetDiscriminatorLossFunction()->ForwardPropagateOnGPU();
    this->SaveLossX();
    this->GetDiscriminatorLossFunction()->BackPropagateOnGPU();
    this->GetDiscriminator()->BackPropagateOnGPU();

    return TRUE;
}

template<typename DTYPE> int BEGAN<DTYPE>::ComputeGradientOfDiscriminatorAboutFakeOnGPU(){
    this->ResetResult();
    this->ResetGradient();
    // this->GetDiscriminator()->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(FAKE);
    this->GetSwitch()->SetSwitchNumber(FAKE);
    this->ForwardPropagateOnGPU();
    // std::cout << ", D : D(G(z)) = " << (*this->GetDiscriminator()->GetResult()->GetResult())[Index5D(this->GetDiscriminator()->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

    // this->MultiplyKOnOutput();
    this->GetDiscriminatorLossFunction()->ForwardPropagateOnGPU();
    this->GetDiscriminatorLossFunction()->BackPropagateOnGPU();
    this->MultiplyKOnGradient();
    this->GetDiscriminator()->BackPropagateOnGPU();

    return TRUE;
}

template<typename DTYPE> float BEGAN<DTYPE>::SaveLossX(){
    return m_LossX = (*(this->GetDiscriminatorLossFunction()->GetResult()))[0];
}

template<typename DTYPE> float BEGAN<DTYPE>::SaveLossG(){
    return m_LossG = (*(this->GetGeneratorLossFunction()->GetResult()))[0];
}

template<typename DTYPE> int BEGAN<DTYPE>::MultiplyKOnOutput(){
    this->GetDiscriminator()->GetResult()->GetResult()->MultiplyScalar(0, m_k);
    return TRUE;
}

template<typename DTYPE> int BEGAN<DTYPE>::MultiplyKOnGradient(){
    this->GetDiscriminator()->GetResult()->GetGradient()->MultiplyScalar(0, m_k);
    return TRUE;
}

template<typename DTYPE> float BEGAN<DTYPE>::UpdateK(){
    printf("m_k = %f, m_LossX = %f, m_LossG = %f, ", m_k, m_LossX, m_LossG);
    m_k = m_k + m_lamda * (m_gamma * m_LossX - m_LossG);
    if(m_k > 1.f) m_k = 1.f;
    if(m_k < 0.f) m_k = 0.f;

    return m_k;
}

template<typename DTYPE> float BEGAN<DTYPE>::ComputeConvergenceMeasure(){
    m_ConvergenceMeasure = m_LossX + std::abs(m_gamma * m_LossX - m_LossG);
    printf("m_ConvergenceMeasure = %f\n", m_ConvergenceMeasure);
    return  m_ConvergenceMeasure;
}

#endif  // BEGAN_H_

//////////////////////////////////////////
