#ifndef __LAYER__
#define __LAYER__    value

#include "Operator_utils.h"

template<typename DTYPE> class Layer : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaExcutableOperator;
    int m_numOfExcutableOperator;

    Operator<DTYPE> *m_pLastOperator;

private:
    int  Alloc();
    void Delete();

public:
    Layer(std::string pName = "No Name");
    virtual ~Layer();

    Operator<DTYPE>                   * SetInput(Operator<DTYPE> *pInput);
    int                                 SetInput(int pNumOfInput, ...);
    int                                 IsInput(Operator<DTYPE> *pOperator);

    int                                 IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>                   * AnalyseGraph(Operator<DTYPE> *pResultOperator);

    Container<Operator<DTYPE> *>      * GetExcutableOperatorContainer();
    int                                 GetNumOfExcutableOperator();

    virtual Tensor<DTYPE>             * GetResult() const;
    virtual Container<Tensor<DTYPE> *>* GetResultContainer();

    virtual Tensor<DTYPE>             * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>* GetGradientContainer();

    virtual Tensor<DTYPE>             * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>* GetDeltaContainer();

    int                                 ForwardPropagate(int pTime = 0, int pThreadNum = 0);
    int                                 BackPropagate(int pTime = 0, int pThreadNum = 0);

    int                                 ResetResult();
    int                                 ResetGradient();

    void                                PrintInformation();

    void                                SetDeviceCPU();
    void                                SetDeviceCPU(int pnumOfThread);

    // int                                 SetResultOnCPU();
    // int                                 SetGradientOnCPU();
#ifdef __CUDNN__
    // int                                 SetResultOnGPU();
    // int                                 SetGradientOnGPU();

    void SetDeviceGPU();
    void SetDeviceGPU(cudnnHandle_t& pCudnnHandle);
    void InitializeAttributeForGPU();

    int  ForwardPropagateOnGPU(int pTime = 0);
    int  BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__
};

#endif  // ifndef __LAYER__
