#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Layer_utils.h"

typedef struct {
    void *m_NN;
    int   m_threadNum;
} ThreadInfo;

template<typename DTYPE> class NeuralNetwork {
private:
#if __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Tensorholder<DTYPE> *> *m_aaTensorholder;
    Container<Layer<DTYPE> *> *m_aaLayer;
    // Parameter

    int m_OperatorDegree;
    int m_TensorholderDegree;

    // 중간에 Loss Function이나 Optimizer가 바뀌는 상황 생각해두기
    LossFunction<DTYPE> *m_aLossFunction;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;
    int m_numOfThread;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    int  Alloc();
    void Delete();

    // =======

    // 추후 직접 변수를 만들지 않은 operator* + operator*의 변환 변수도 자동으로 할당될 수 있도록 Operator와 NN class를 수정해야 한다.
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddTensorholder(Tensorholder<DTYPE> *pTensorholder);
    Tensorholder<DTYPE>* AddParameter(Tensorholder<DTYPE> *pTensorholder);
    // Operator<DTYPE>    * AddLayer(Layer<DTYPE> *pLayer);

    LossFunction<DTYPE>* SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    Operator<DTYPE>                 * GetResultOperator();
    Operator<DTYPE>                 * GetResult();
    Container<Operator<DTYPE> *>    * GetOperatorContainer();


    Container<Tensorholder<DTYPE> *>* GetTensorholder();
    Container<Tensorholder<DTYPE> *>* GetParameter();

    LossFunction<DTYPE>             * GetLossFunction();
    Optimizer<DTYPE>                * GetOptimizer();

    // =======
    float                             GetAccuracy();
    int                               GetMaxIndex(Tensor<DTYPE> *data, int ba, int numOfClass);
    float                             GetLoss();

    // =======
    int                               ForwardPropagate(int pTime = 0);
    int                               BackPropagate(int pTime = 0);

    static void                     * ForwardPropagateForThread(void *param);
    static void                     * BackPropagateForThread(void *param);

#if __CUDNN__
    int                               ForwardPropagateOnGPU(int pTime = 0);
    int                               BackPropagateOnGPU(int pTime = 0);
#endif  // __CUDNN__

    // =======
    int Training();
    int Testing();

    int TrainingOnCPU();
    int TestingOnCPU();

    int TrainingOnMultiThread();  // Multi Threading
    int TestingOnMultiThread();  // Multi Threading

    int TrainingOnGPU();
    int TestingOnGPU();

    // int                               TrainingOnMultiProcess(); // Multi Processing
    // int                               TestingOnMultiProcess(); // Multi Processing

    // int                               TrainingOnMultiGPU();
    // int                               TestingOnMultiGPU();

    // ============
    void             SetModeTraining();
    void             SetModeAccumulating();
    void             SetModeInferencing();

#if __CUDNN__
    void             SetDeviceGPU();
#endif  // __CUDNN__

    void             SetDeviceCPU();
    void             SetDeviceCPU(int pNumOfThread);

    // =======
    int              CreateGraph();
    void             PrintGraphInformation();

    // reset value
    int              ResetOperatorResult();
    int              ResetOperatorGradient();

    int              ResetLossFunctionResult();
    int              ResetLossFunctionGradient();

    int              ResetParameterGradient();

    // debug
    Operator<DTYPE>* SerchOperator(std::string pName);
};

#endif  // NEURALNETWORK_H_
