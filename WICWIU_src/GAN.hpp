#ifndef GAN_H_
#define GAN_H_

#include "NeuralNetwork.hpp"

template<typename DTYPE> class GAN : public NeuralNetwork<DTYPE> {
private:
    NeuralNetwork<DTYPE> *m_pGenerator;
    NeuralNetwork<DTYPE> *m_pDiscriminator;

    Tensorholder<DTYPE> *m_pLabel;
    Switch<DTYPE> *m_pSwitch;

    LossFunction<DTYPE> *m_pGeneratorLossFunction;
    LossFunction<DTYPE> *m_pDiscriminatorLossFunction;

private:
    int AllocLabelOnCPU(float plabelValue);

#ifdef __CUDNN__
    int AllocLabelOnGPU(float plabelValue);
#endif

public:
    GAN();
    virtual ~GAN();

    int                                 AllocLabel(float plabelValue);

    NeuralNetwork<DTYPE>*               SetGenerator(NeuralNetwork<DTYPE> *pGen);
    NeuralNetwork<DTYPE>*               SetDiscriminator(NeuralNetwork<DTYPE> *pDiscLoss);

    Tensorholder<DTYPE>*                SetLabel(Tensorholder<DTYPE> *pLabel);
    Switch<DTYPE>*                      SetSwitch(Switch<DTYPE> *pSwitch);

    void                                SetGANLossFunctions(LossFunction<DTYPE> *pGenLoss, LossFunction<DTYPE> *pDiscLoss);
    LossFunction<DTYPE>*                SetGeneratorLossFunction(LossFunction<DTYPE> *pGenLoss);
    LossFunction<DTYPE>*                SetDiscriminatorLossFunction(LossFunction<DTYPE> *pDiscLoss);

    void                                SetGANOptimizers(Optimizer<DTYPE> *pGenOpt, Optimizer<DTYPE> *pDiscOpt);
    Optimizer<DTYPE>*                   SetGeneratorOptimizer(Optimizer<DTYPE> *pGenOpt);
    Optimizer<DTYPE>*                   SetDiscriminatorOptimizer(Optimizer<DTYPE> *pDiscOpt);


    NeuralNetwork<DTYPE>*               GetGenerator();
    NeuralNetwork<DTYPE>*               GetDiscriminator();

    Tensorholder<DTYPE>*                GetLabel();
    Switch<DTYPE>*                      GetSwitch();

    LossFunction<DTYPE>*                GetGeneratorLossFunction();
    LossFunction<DTYPE>*                GetDiscriminatorLossFunction();

    Optimizer<DTYPE>*                   GetGeneratorOptimizer();
    Optimizer<DTYPE>*                   GetDiscriminatorOptimizer();


    int                                 TrainGenerator();
    int                                 TrainDiscriminator();

    int                                 Generate();

    virtual int                         TrainGeneratorOnCPU();
    virtual int                         TrainDiscriminatorOnCPU();

    virtual int                         ComputeGradientOfDiscriminatorAboutRealOnCPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutFakeOnCPU();

    int                                 GenerateOnCPU();

    virtual int                         TrainGeneratorOnGPU();
    virtual int                         TrainDiscriminatorOnGPU();

    virtual int                         ComputeGradientOfDiscriminatorAboutRealOnGPU();
    virtual int                         ComputeGradientOfDiscriminatorAboutFakeOnGPU();

    int                                 GenerateOnGPU();

    int                                 ResetParameterGradient();

    int                                 ResetGeneratorLossFunctionResult();
    int                                 ResetGeneratorLossFunctionGradient();

    int                                 ResetDiscriminatorLossFunctionResult();
    int                                 ResetDiscriminatorLossFunctionGradient();

    void                                Clip(float min, float max);

#ifdef __CUDNN__
    void                                SetDeviceGPUOnGAN(unsigned int idOfDevice);
#endif  // __CUDNN__
};

template<typename DTYPE> int GAN<DTYPE>::AllocLabelOnCPU(float plabelValue){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::AllocLabelOnCPU(int plabel)" << '\n';
    #endif  // __DEBUG__

    int m_timesize = m_pLabel->GetResult()->GetDim(0);
    int m_batchsize = m_pLabel->GetResult()->GetDim(1);
    int m_channelsize = m_pLabel->GetResult()->GetDim(2);
    int m_rowsize = m_pLabel->GetResult()->GetDim(3);
    int m_colsize = m_pLabel->GetResult()->GetDim(4);

    m_pLabel->FeedTensor(Tensor<DTYPE>::Constants(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize, plabelValue));

    return true;
}

#ifdef __CUDNN__
template<typename DTYPE> int GAN<DTYPE>::AllocLabelOnGPU(float plabelValue){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::AllocLabelOnGPU(int plabel)" << '\n';
    #endif  // __DEBUG__

    int m_timesize = m_pLabel->GetResult()->GetDim(0);
    int m_batchsize = m_pLabel->GetResult()->GetDim(1);
    int m_channelsize = m_pLabel->GetResult()->GetDim(2);
    int m_rowsize = m_pLabel->GetResult()->GetDim(3);
    int m_colsize = m_pLabel->GetResult()->GetDim(4);

    m_pLabel->FeedTensor(Tensor<DTYPE>::Constants(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize, plabelValue));
    m_pLabel->GetTensor()->SetDeviceGPU(this->GetDeviceID());

    return true;
}
#endif  // __CUDNN__


template<typename DTYPE> GAN<DTYPE>::GAN() : NeuralNetwork<DTYPE>() {
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::GAN()" << '\n';
    #endif  // __DEBUG__

    m_pGenerator = NULL;
    m_pDiscriminator = NULL;

    m_pLabel = NULL;
    m_pSwitch = NULL;

    m_pGeneratorLossFunction = NULL;
    m_pDiscriminatorLossFunction = NULL;

}

template<typename DTYPE> GAN<DTYPE>::~GAN(){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::~GAN()" << '\n';
    #endif  // __DEBUG__
}

template<typename DTYPE> int GAN<DTYPE>::AllocLabel(float plabelValue){
    if(this->GetDevice() == CPU) {
        this->AllocLabelOnCPU(plabelValue);
    } else if(this->GetDevice() == GPU) {
        this->AllocLabelOnGPU(plabelValue);
    } else return FALSE;
}

// Setter
template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetGenerator(NeuralNetwork<DTYPE> *pGen){
    m_pGenerator = pGen;
    return m_pGenerator;
}

template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetDiscriminator(NeuralNetwork<DTYPE> *pDisc){
    m_pDiscriminator = pDisc;
    return m_pDiscriminator;
}

template<typename DTYPE> Switch<DTYPE>* GAN<DTYPE>::SetSwitch(Switch<DTYPE> *pSwitch){
    m_pSwitch = pSwitch;
    return m_pSwitch;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::SetLabel(Tensorholder<DTYPE> *pLabel){
    m_pLabel = pLabel;
    return m_pLabel;
}

template<typename DTYPE> void GAN<DTYPE>::SetGANLossFunctions(LossFunction<DTYPE> *pGenLoss, LossFunction<DTYPE> *pDiscLoss){
    SetGeneratorLossFunction(pGenLoss);
    SetDiscriminatorLossFunction(pDiscLoss);
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::SetGeneratorLossFunction(LossFunction<DTYPE> *pGenLoss){
    m_pGeneratorLossFunction = pGenLoss;
    return m_pGenerator->SetLossFunction(pGenLoss);
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::SetDiscriminatorLossFunction(LossFunction<DTYPE> *pDiscLoss){
    m_pDiscriminatorLossFunction = pDiscLoss;
    return m_pDiscriminator->SetLossFunction(pDiscLoss);
}

template<typename DTYPE> void GAN<DTYPE>::SetGANOptimizers(Optimizer<DTYPE> *pGenOpt, Optimizer<DTYPE> *pDiscOpt){
    SetGeneratorOptimizer(pGenOpt);
    SetDiscriminatorOptimizer(pDiscOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetGeneratorOptimizer(Optimizer<DTYPE> *pGenOpt){
    return m_pGenerator->SetOptimizer(pGenOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetDiscriminatorOptimizer(Optimizer<DTYPE> *pDiscOpt){
    return m_pDiscriminator->SetOptimizer(pDiscOpt);
}

// Getter
template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::GetGenerator(){
    return m_pGenerator;
}

template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::GetDiscriminator(){
    return m_pDiscriminator;
}

template<typename DTYPE> Switch<DTYPE>* GAN<DTYPE>::GetSwitch(){
    return m_pSwitch;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::GetLabel(){
    return m_pLabel;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::GetGeneratorLossFunction(){
    return m_pGeneratorLossFunction;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::GetDiscriminatorLossFunction(){
    return m_pDiscriminatorLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetGeneratorOptimizer(){
    return m_pGenerator->GetOptimizer();
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetDiscriminatorOptimizer(){
    return m_pDiscriminator->GetOptimizer();
}

template<typename DTYPE> int GAN<DTYPE>::TrainGenerator(){
    if(this->GetDevice() == CPU) {
        this->TrainGeneratorOnCPU();
    } else if(this->GetDevice() == GPU) {
        this->TrainGeneratorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminator(){
    if(this->GetDevice() == CPU) {
        this->TrainDiscriminatorOnCPU();
    } else if(this->GetDevice() == GPU) {
        this->TrainDiscriminatorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::Generate() {
  if(this->GetDevice() == CPU) {
      this->GenerateOnCPU();
  } else if(this->GetDevice() == GPU) {
      this->GenerateOnGPU();
  } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnCPU(){
    this->ResetResult();
    this->ResetGradient();
    this->ResetGeneratorLossFunctionResult();
    this->ResetGeneratorLossFunctionGradient();

    m_pSwitch->SetSwitchNumber(FAKE);
    this->ForwardPropagate();
    m_pGeneratorLossFunction->ForwardPropagate();
    m_pGeneratorLossFunction->BackPropagate();
    this->BackPropagate();

    this->GetGeneratorOptimizer()->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnCPU(){
    this->ComputeGradientOfDiscriminatorAboutRealOnCPU();
    this->GetDiscriminatorOptimizer()->UpdateParameter();
    this->ComputeGradientOfDiscriminatorAboutFakeOnCPU();
    this->GetDiscriminatorOptimizer()->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ComputeGradientOfDiscriminatorAboutRealOnCPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REAL);
    m_pSwitch->SetSwitchNumber(REAL);
    m_pSwitch->ForwardPropagate();
    m_pDiscriminator->ForwardPropagate();
    m_pDiscriminatorLossFunction->ForwardPropagate();
    m_pDiscriminatorLossFunction->BackPropagate();
    m_pDiscriminator->BackPropagate();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ComputeGradientOfDiscriminatorAboutFakeOnCPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();
    // this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(FAKE);
    m_pSwitch->SetSwitchNumber(FAKE);
    this->ForwardPropagate();
    m_pDiscriminatorLossFunction->ForwardPropagate();
    m_pDiscriminatorLossFunction->BackPropagate();
    m_pDiscriminator->BackPropagate();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::GenerateOnCPU(){
    m_pGenerator->ResetResult();
    m_pGenerator->ForwardPropagate();

    return TRUE;
}


template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ResetGradient();
        this->ResetGeneratorLossFunctionResult();
        this->ResetGeneratorLossFunctionGradient();

        m_pSwitch->SetSwitchNumber(FAKE);
        this->ForwardPropagateOnGPU();
        m_pGeneratorLossFunction->ForwardPropagateOnGPU();
        m_pGeneratorLossFunction->BackPropagateOnGPU();
        this->BackPropagateOnGPU();

        this->GetGeneratorOptimizer()->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnGPU(){
    #ifdef __CUDNN__
        this->ComputeGradientOfDiscriminatorAboutRealOnGPU();
        this->GetDiscriminatorOptimizer()->UpdateParameterOnGPU();
        this->ComputeGradientOfDiscriminatorAboutFakeOnGPU();
        this->GetDiscriminatorOptimizer()->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ComputeGradientOfDiscriminatorAboutRealOnGPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REAL);
    m_pSwitch->SetSwitchNumber(REAL);
    m_pSwitch->ForwardPropagateOnGPU();
    m_pDiscriminator->ForwardPropagateOnGPU();
    m_pDiscriminatorLossFunction->ForwardPropagateOnGPU();
    m_pDiscriminatorLossFunction->BackPropagateOnGPU();
    m_pDiscriminator->BackPropagateOnGPU();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ComputeGradientOfDiscriminatorAboutFakeOnGPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();
    // this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(FAKE);
    m_pSwitch->SetSwitchNumber(FAKE);
    this->ForwardPropagateOnGPU();
    m_pDiscriminatorLossFunction->ForwardPropagateOnGPU();
    m_pDiscriminatorLossFunction->BackPropagateOnGPU();
    m_pDiscriminator->BackPropagateOnGPU();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::GenerateOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ForwardPropagateOnGPU();
    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

        return TRUE;
}

// Need to override
template<typename DTYPE> int GAN<DTYPE>::ResetParameterGradient(){
    this->GetGeneratorOptimizer()->ResetParameterGradient();
    this->GetDiscriminatorOptimizer()->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionResult(){
    m_pGeneratorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionGradient(){
    m_pGeneratorLossFunction->ResetGradient();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionResult(){
    m_pDiscriminatorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionGradient(){
    m_pDiscriminatorLossFunction->ResetGradient();
    return TRUE;
}

template<typename DTYPE> void GAN<DTYPE>::Clip(float min, float max){
    this->GetDiscriminator()->GetResult()->GetResult()->Clip(min, max);
}

#ifdef __CUDNN__

template<typename DTYPE> void GAN<DTYPE>::SetDeviceGPUOnGAN(unsigned int idOfDevice) {
    this->SetDeviceGPU(idOfDevice);
    m_pGenerator->SetDeviceGPU(idOfDevice);
    m_pDiscriminator->SetDeviceGPU(idOfDevice);
}

#endif  // __CUDNN__

#endif  // GAN_H_
