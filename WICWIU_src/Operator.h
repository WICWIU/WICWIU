#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.h"
#include "Container.h"

enum Mode {
    TRAINING,
    ACCUMULATING,
    INFERENCING,
};

template<typename DTYPE> class Operator {
private:
    Container<Operator<DTYPE> *> *m_apOutput;
    Container<Operator<DTYPE> *> *m_apInput;
    Container<Tensor<DTYPE> *> *m_aaResult;
    Container<Tensor<DTYPE> *> *m_aaGradient;
    std::string m_name;
    Device m_Device;
    int m_idOfDevice = -1;
    Mode m_Mode;
    int m_isParameter;
    int m_isTrainable;

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
#endif  // __CUDNN__

private:
    int  Alloc();
    int  Alloc(int numInput, ...);
    void Delete();

    int  AddInputEdge(Operator<DTYPE> *pInput);
    int  AddOutputEdge(Operator<DTYPE> *pOutput);


#ifdef __CUDNN__

#endif  // __CUDNN__

public:
    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    virtual ~Operator();

    int                                   AddEdgebetweenOperators(Operator<DTYPE> *pInput);
    int                                   AddEdgebetweenOperators(int numInput, ...);
    int                                   AddResult(Tensor<DTYPE> *pTensor);
    int                                   AddGradient(Tensor<DTYPE> *pTensor);
    int                                   AddDelta(Tensor<DTYPE> *pTensor);
    int                                   SetResult(Tensor<DTYPE> *pTensor);     // 0 or 1 일 때만 진행 가능
    int                                   SetGradient(Tensor<DTYPE> *pTensor);
    int                                   SetDelta(Tensor<DTYPE> *pTensor);

    int                                   SetDevice(Device pDevice);
    int                                   SetDeviceID(unsigned int idOfDevice);

    int                                   SetIsTensorholder(int pIsParameter);
    int                                   SetIsTrainable(int pIsTrainable);

    virtual int                           SetModeTraining();
    virtual int                           SetModeAccumulating();
    virtual int                           SetModeInferencing();

    virtual Operator<DTYPE>            ** GetOutput();
    virtual Container<Operator<DTYPE> *>* GetOutputContainer();
    virtual Operator<DTYPE>            ** GetInput();
    virtual Container<Operator<DTYPE> *>* GetInputContainer();
    virtual Tensor<DTYPE>               * GetResult() const;
    virtual Container<Tensor<DTYPE> *>  * GetResultContainer();
    virtual Tensor<DTYPE>               * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>  * GetGradientContainer();
    virtual Tensor<DTYPE>               * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>  * GetDeltaContainer();

    std::string                           GetName() const;
    virtual Device                        GetDevice();
    virtual int                           GetDeviceID();
    int                                   GetIsTensorholder();
    int                                   GetIsTrainable();

    virtual int                           ForwardPropagate(int pTime = 0);
    virtual int                           BackPropagate(int pTime = 0);

    // reset value
    virtual int                           ResetResult();
    virtual int                           ResetGradient();

    virtual void                          PrintInformation();

    virtual void                          SetDeviceCPU();

    virtual int                           SetResultOnCPU();
    virtual int                           SetGradientOnCPU();
#ifdef __CUDNN__
    int                                   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    virtual int                           SetResultOnGPU(unsigned int idOfDevice);
    virtual int                           SetGradientOnGPU(unsigned int idOfDevice);

    // virtual void                          SetDeviceGPU(unsigned int idOfDevice);
    virtual void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void                          InitializeAttributeForGPU(unsigned int idOfDevice);

    cudnnHandle_t& GetCudnnHandle();

    virtual int    ForwardPropagateOnGPU(int pTime = 0);
    virtual int    BackPropagateOnGPU(int pTime = 0);


#endif  // if __CUDNN__
};

#endif  // OPERATOR_H_
