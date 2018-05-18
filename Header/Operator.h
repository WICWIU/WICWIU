#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.h"
#include "Container.h"

enum Mode {
    TRAINING,
    ACCUMULATING,
    INFERENCING,
};

enum Device {
    CPU,
    GPU,
};

template<typename DTYPE> class Operator {
private:
    Container<Tensor<DTYPE> *> *m_aaResult;
    Container<Tensor<DTYPE> *> *m_aaGradient;

    Container<Operator<DTYPE> *> *m_apOutput;
    Container<Operator<DTYPE> *> *m_apInput;

    int m_OutputDegree;
    int m_InputDegree;

    int m_currentOutputDegree;
    int m_currentInputDegree;

    int m_numOfParameter;

    std::string m_name;

    Device m_Device;

    int m_isTensorholder;
    int m_isTrainable;

    int m_numOfThread;

public:
#if __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
    cudnnHandle_t& GetCudnnHandle();
    virtual void   InitializeAttributeForGPU();
    virtual void   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    void           cudnnResize(int size, float *data);
#endif  // if __CUDNN__

    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    virtual ~Operator();

    virtual int                         Alloc();
    virtual int                         Alloc(int numInput, ...);
    virtual void                        Delete();

    void                                SetResult(Tensor<DTYPE> *pTensor);
    void                                AddResult(Tensor<DTYPE> *pTensor);

    void                                SetGradient(Tensor<DTYPE> *pTensor);
    void                                AddGradient(Tensor<DTYPE> *pTensor);

    void                                SetDelta(Tensor<DTYPE> *pTensor);
    void                                AddDelta(Tensor<DTYPE> *pTensor);

    void                                IncreaseCurrentOutputDegree();
    void                                IncreaseCurrentInputDegree();

    int                                 _AddInputEdge(Operator<DTYPE> *pInput);
    int                                 _AddOutputEdge(Operator<DTYPE> *pOutput);
    void                                AddEdgebetweenOperators(Operator<DTYPE> *pInput);

    virtual Tensor<DTYPE>             * GetResult() const;
    virtual Container<Tensor<DTYPE> *>* GetResultContainer();

    virtual Tensor<DTYPE>             * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>* GetGradientContainer();

    virtual Tensor<DTYPE>             * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>* GetDeltaContainer();

    Operator<DTYPE>                  ** GetOutput();
    Container<Operator<DTYPE> *>      * GetOutputContainer();

    Operator<DTYPE>                  ** GetInput();
    Container<Operator<DTYPE> *>      * GetInputContainer();

    int                                 GetOutputDegree() const;
    int                                 GetInputDegree() const;

    int                                 GetCurrentOutputDegree() const;
    int                                 GetCurrentInputDegree() const;

    std::string                         GetName() const;

    // Operator<DTYPE>             * Concatenate(Operator<DTYPE> *src, Operator<DTYPE> *dst, int axis = 0);

    // For Propagate
    virtual int  ForwardPropagate();
    virtual int  ForwardPropagate(int pTime, int pThreadNum);

    // For BackPropagate
    virtual int  BackPropagate();
    virtual int  BackPropagate(int pTime, int pThreadNum);

    // reset value
    virtual int  ResetResult();
    virtual int  ResetGradient();

    virtual void SetModeTraining();
    virtual void SetModeAccumulating();
    virtual void SetModeInferencing();


    virtual void SetDeviceCPU();
    virtual void SetDeviceCPU(int pNumOfThread);
#ifdef __CUDNN__
    virtual void SetDeviceGPU();

#endif  // if __CUDNN__

    virtual Device GetDevice() {
        return m_Device;
    }

    int GetNumOfThread() {
        return m_numOfThread;
    }

    virtual int GetNumOfParameter() {
        return 0;
    }

    virtual Container<Tensorholder<DTYPE> *>* GetParameterContainer() {
        return NULL;
    }

    virtual Tensorholder<DTYPE>* PopParameter() {
        return NULL;
    }

    virtual void PrintInformation();
};

#endif  // OPERATOR_H_
