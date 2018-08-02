#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.h"

template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Operator<DTYPE> *> *m_apExcutableOperator;
    Container<Operator<DTYPE> *> *m_apInput;
    Container<Operator<DTYPE> *> *m_apParameter;

    int m_Operatordegree;
    int m_ExcutableOperatorDegree;
    int m_InputDegree;
    int m_ParameterDegree;

    LossFunction<DTYPE> *m_aLossFunction;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__

private:
    int  Alloc();
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
#endif  // if __CUDNN__

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    Operator<DTYPE>             * SetInput(Operator<DTYPE> *pInput);
    int                           SetInput(int pNumOfInput, ...);
    int                           IsInput(Operator<DTYPE> *pOperator);

    int                           IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>             * AnalyzeGraph(Operator<DTYPE> *pResultOperator);
    LossFunction<DTYPE>         * SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>            * SetOptimizer(Optimizer<DTYPE> *pOptimizer);
    int                           FeedInputTensor(int pNumOfInput, ...);
    // =======

    Container<Operator<DTYPE> *>* GetInputContainer();

    Operator<DTYPE>             * GetResultOperator();
    Operator<DTYPE>             * GetResult();

    Container<Operator<DTYPE> *>* GetExcutableOperatorContainer();

    Container<Operator<DTYPE> *>* GetParameterContainer();
    Container<Operator<DTYPE> *>* GetParameter();

    LossFunction<DTYPE>         * GetLossFunction();

    Optimizer<DTYPE>            * GetOptimizer();

    int                           ForwardPropagate(int pTime = 0);
    int                           BackPropagate(int pTime = 0);

    void                          SetDeviceCPU();

    void                          SetModeTraining();
    void                          SetModeAccumulating();
    void                          SetModeInferencing();

    int                           Training();
    int                           Testing();

    int                           TrainingOnCPU();
    int                           TestingOnCPU();

    int                           TrainingOnGPU();
    int                           TestingOnGPU();

    float                         GetAccuracy(int numOfClass = 10);
    int                           GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass);
    float                         GetLoss();

    void                          PrintGraphInformation();

    int                           ResetOperatorResult();
    int                           ResetOperatorGradient();

    int                           ResetLossFunctionResult();
    int                           ResetLossFunctionGradient();

    int                           ResetParameterGradient();

    Operator<DTYPE>             * SerchOperator(std::string pName);

#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);

    void                          SetDeviceGPU();
#endif  // __CUDNN__
};

#endif  // NEURALNETWORK_H_
