#ifndef LossFunction_H_
#define LossFunction_H_

#include "Module_utils.hpp"

/*!
@class LossFunction 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와 레이블 값을 비교해 손실 함수를 계산한다.
*/
template <typename DTYPE>
class LossFunction
{
private:
    Tensor<DTYPE>* m_aResult;
    ///< LossFunction에서 얻어진 결과 값을 저장하는 Tensor에 대한 포인터
    Tensor<DTYPE>* m_aGradient;
    ///< LossFunction에서 얻어진 결과 값의 Gradient를 저장하는 Tensor에 대한 포인터

    Operator<DTYPE>* m_pInputOperator;
    ///< LossFunction의 Input에 해당하는 Operator, 즉 NeuralNetwork의 Output에 해당하는 Operator에
    ///< 대한 포인터
    Tensor<DTYPE>* m_pInputTensor;
    ///< NeuralNetwork의 Output에 해당하는 Operator의 Result Tensor에 대한 포인터

    Operator<DTYPE>* m_pLabel;
    ///< 학습 데이터에 대한 Label 값에 대한 포인터

    std::string m_name;
    ///< LossFunction의 이름을 저장하는 string

    Device m_Device;
    ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    int m_idOfDevice;
    ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
    ///< cudnn handler
#endif // if __CUDNN__

public:
    LossFunction(std::string pName = "NO NAME");
    LossFunction(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel,
                 std::string pName = "NO NAME");

    virtual ~LossFunction();

    virtual int Alloc(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel);
    virtual void Delete();

    void SetResult(Tensor<DTYPE>* pTensor);
    void SetGradient(Tensor<DTYPE>* pTensor);

    Tensor<DTYPE>* GetResult() const;
    Tensor<DTYPE>* GetGradient() const;
    Operator<DTYPE>* GetOperator() const;
    Tensor<DTYPE>* GetTensor() const;
    Operator<DTYPE>* GetLabel() const;
    std::string GetName() const;
    virtual Device GetDevice();
    virtual int GetDeviceID();

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagate(int pTime = 0);

#ifdef __CUDNN__
    virtual Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0);
#endif // if __CUDNN__

    DTYPE& operator[](unsigned int index);

    virtual void SetDeviceCPU();
#ifdef __CUDNN__

    // Setting Supporter
    virtual int SetResultOnCPU();
    virtual int SetGradientOnCPU();

    // virtual void   SetDeviceGPU(unsigned int idOfDevice);
    virtual void SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void InitializeAttributeForGPU(unsigned int idOfDevice);

    cudnnHandle_t& GetCudnnHandle();

    // Setting Supporter
    virtual int SetResultOnGPU(unsigned int idOfDevice);
    virtual int SetGradientOnGPU(unsigned int idOfDevice);

#endif // if __CUDNN__

    // reset value
    int ResetResult();
    int ResetGradient();
};

/*!
@brief LossFunction 클래스 생성자
@details LossFunction의 멤버 변수 포인터들을 NULL값으로 초기화하고, 매개변수로 받은 스트링을
m_name에 저장하고, m_Device를 CPU로 초기화한다.
@param pName m_name에 할당할 LossFunction의 이름, 값을 전달하지 않을 시 "NO NAME"으로 초기화 됨
@return 없음
*/
template <typename DTYPE>
LossFunction<DTYPE>::LossFunction(std::string pName)
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
#endif // __DEBUG__
    m_aResult = NULL;
    m_aGradient = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor = NULL;
    m_pLabel = NULL;
    m_name = pName;
    m_Device = CPU;
    m_idOfDevice = -1;
}

/*!
@brief LossFunction 클래스 생성자
@details LossFunction의 멤버 변수 포인터들을 NULL값으로 초기화하고, 매개변수로 받은 스트링을
m_name에 저장하고, m_Device를 CPU로 초기화한다.
@details pOperator와 pLabel을 매개변수로 LossFunction<DTYPE>::Alloc(Operator<DTYPE> *pOperator,
Operator<DTYPE> *pLabel) 메소드를 호출한다.
@param pOperator Alloc 메소드의 매개변수로 전달할 LossFunction의 입력에 해당하는 Operator
@param pLabel Alloc 메소드의 매개변수로 전달할 LossFunction의 입력에 해당하는 레이블
@param pName m_name에 할당할 LossFunction의 이름, 값을 전달하지 않을 시 "NO NAME"으로 초기화 됨
@return 없음
*/
template <typename DTYPE>
LossFunction<DTYPE>::LossFunction(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel,
                                  std::string pName)
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
#endif // __DEBUG__
    m_aResult = NULL;
    m_aGradient = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor = NULL;
    m_pLabel = NULL;
    m_name = pName;
    m_Device = CPU;
    m_idOfDevice = -1;
    Alloc(pOperator, pLabel);
}

/*!
@brief LossFunction 클래스 소멸자
@details LossFunction<DTYPE>::Delete() 메소드를 호출하고 클래스를 소멸시킨다.
@return 없음
*/
template <typename DTYPE>
LossFunction<DTYPE>::~LossFunction()
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::~LossFunction()" << '\n';
#endif // __DEBUG__
    this->Delete();
}

/*!
@brief LossFunction의 입력과 레이블을 지정하는 메소드
@details 매개변수로 전달받은 Operator와 Operator의 Result 포인터 값과 레이블 값을 저장한다.
@param pOperator LossFunction의 입력이 되는 Operator
@param plabel LossFunction의 입력이 되는 레이블
@return TRUE
*/
template <typename DTYPE>
int LossFunction<DTYPE>::Alloc(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel)
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
#endif // __DEBUG__

    m_pInputOperator = pOperator;
    m_pInputTensor = m_pInputOperator->GetResult();

    m_pLabel = pLabel;
    return TRUE;
}

/*!
@brief 동적으로 할당받은 LossFunction의 멤버 변수들을 할당 해제하는 메소드
@details Result와 Gradient에 해당하는 Tensor들의 메모리를 할당 해제한다.
@return 없음
*/
template <typename DTYPE>
void LossFunction<DTYPE>::Delete()
{
    if (m_aResult)
    {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient)
    {
        delete m_aGradient;
        m_aGradient = NULL;
    }
}

template <typename DTYPE>
void LossFunction<DTYPE>::SetResult(Tensor<DTYPE>* pTensor)
{
    m_aResult = pTensor;
}

template <typename DTYPE>
void LossFunction<DTYPE>::SetGradient(Tensor<DTYPE>* pTensor)
{
    m_aGradient = pTensor;
}

template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::GetResult() const
{
    return m_aResult;
}

template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::GetGradient() const
{
    return m_aGradient;
}

template <typename DTYPE>
Operator<DTYPE>* LossFunction<DTYPE>::GetOperator() const
{
    return m_pInputOperator;
}

template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::GetTensor() const
{
    return m_pInputTensor;
}

template <typename DTYPE>
Operator<DTYPE>* LossFunction<DTYPE>::GetLabel() const
{
    return m_pLabel;
}

template <typename DTYPE>
std::string LossFunction<DTYPE>::GetName() const
{
    return m_name;
}

template <typename DTYPE>
Device LossFunction<DTYPE>::GetDevice()
{
    return m_Device;
}

template <typename DTYPE>
int LossFunction<DTYPE>::GetDeviceID()
{
    return m_idOfDevice;
}

/*!
@brief LossFunction의 순전파를 수행하는 메소드
@param pTime 학습 데이터 텐서의 Time 인덱스, 값을 전달하지 않을 시 0으로 초기화 됨
@return NULL
*/
template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::ForwardPropagate(int pTime)
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::ForwardPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
#endif // __DEBUG__
    return NULL;
}

/*!
@brief LossFunction의 역전파를 수행하는 메소드
@param pTime 학습 데이터 텐서의 Time 인덱스, 값을 전달하지 않을 시 0으로 초기화 됨
@return NULL
*/
template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::BackPropagate(int pTime)
{
#ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::BackPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
#endif // __DEBUG__
    return NULL;
}

#ifdef __CUDNN__

/*!
@brief GPU에서의 LossFunction의 순전파를 수행하는 메소드.
@param pTime
@return NULL
*/
template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime)
{
#if __DEBUG__
    std::cout << this->GetName() << '\n';
#endif // __DEBUG__
    return NULL;
}

/*!
@brief GPU에서의 LossFunction의 순전파를 수행하는 메소드.
@param pTime
@return NULL
*/
template <typename DTYPE>
Tensor<DTYPE>* LossFunction<DTYPE>::BackPropagateOnGPU(int pTime)
{
#if __DEBUG__
    std::cout << this->GetName() << '\n';
#endif // __DEBUG__
    return NULL;
}

#endif // __CUDNN__

/*!
@brief [] 연산자 오버로딩
@details 매개변수로 전달받은 index 값 매개변수로 전달하여 Result 텐서에서 []연산자 메소드를
호출한다.
@param index Tensor의 [] 연산자 메소드에 매개변수로 전달할 인덱스 값
@return (*m_aResult)[index]
@see Tensor<DTYPE>::operator[](unsigned int index)
*/
template <typename DTYPE>
DTYPE& LossFunction<DTYPE>::operator[](unsigned int index)
{
    return (*m_aResult)[index];
}

template <typename DTYPE>
void LossFunction<DTYPE>::SetDeviceCPU()
{
    m_Device = CPU;

#ifdef __CUDNN__
    this->SetResultOnCPU();
    this->SetGradientOnCPU();
#endif // __CUDNN__
}

#ifdef __CUDNN__
/*!
@brief Result 텐서의 Device 멤버 변수를 CPU로 설정하는 메소드
@details Result 텐서가 정상적으로 할당되어 있는 경우, Result 텐서의 Device 멤버 변수를 CPU로
설정한다.
@return TRUE
*/
template <typename DTYPE>
int LossFunction<DTYPE>::SetResultOnCPU()
{
    if (m_aResult)
        m_aResult->SetDeviceCPU();

    return TRUE;
}

/*!
@brief Gradient 텐서의 Device 멤버 변수를 CPU로 설정하는 메소드
@details Gradient 텐서가 정상적으로 할당되어 있는 경우, Result 텐서의 Device 멤버 변수를 CPU로
설정한다.
@return TRUE
*/
template <typename DTYPE>
int LossFunction<DTYPE>::SetGradientOnCPU()
{
    if (m_aGradient)
        m_aGradient->SetDeviceCPU();

    return TRUE;
}

// template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// m_Device = GPU;
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

/*!
@brief LossFunction 클래스의 device 맴버 변수를 GPU로 변경한다.
@details LossFunction의 Result와 Gradient의 Device를 GPU로 변경한다.
@param pCudnnHandle cudnn 라이브러리를 가리키는 구조체 포인터.
@param idOfDevice 사용하고자 하는 GPU번호
*/
template <typename DTYPE>
void LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice)
{
    checkCudaErrors(cudaSetDevice(idOfDevice));
    m_Device = GPU;
    m_idOfDevice = idOfDevice;
    m_pCudnnHandle = pCudnnHandle;
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

template <typename DTYPE>
void LossFunction<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice)
{
}

/*!
@brief m_aResult의 device 맴버 변수를 GPU로 변경한다.
@param idOfDevice 사용하고자 하는 GPU번호
@return 성공 시 TRUE
*/
template <typename DTYPE>
int LossFunction<DTYPE>::SetResultOnGPU(unsigned int idOfDevice)
{
    if (m_aResult)
        m_aResult->SetDeviceGPU(idOfDevice);

    return TRUE;
}

/*!
@brief m_aGradient의 device 맴버 변수를 GPU로 변경한다.
@param idOfDevice 사용하고자 하는 GPU번호
@return 성공 시 TRUE.
*/
template <typename DTYPE>
int LossFunction<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice)
{
    if (m_aGradient)
        m_aGradient->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template <typename DTYPE>
cudnnHandle_t& LossFunction<DTYPE>::GetCudnnHandle()
{
    return m_pCudnnHandle;
}

#endif // __CUDNN__

/*!
@brief Result 텐서의 ELement를 0으로 초기화하는 메소드
@details Result 텐서의 Device 멤버 변수가 CPU인 경우 CPU 메모리에서 초기화하고, CPU인 경우 GPU
메모리에서 초기화한다.
@return Result 텐서의 Device 멤버 변수가 Invalid한 값을 가지고 있는 경우 FALSE를 그 외의 경우 TRUE를
반환한다.
*/
template <typename DTYPE>
int LossFunction<DTYPE>::ResetResult()
{
    if (m_Device == CPU)
    {
        if (m_aResult)
            m_aResult->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU)
    {
        if (m_aResult)
            m_aResult->Reset(this->GetCudnnHandle());
    }
#endif // if __CUDNN__

    else
        return FALSE;

    return TRUE;
}

/*!
@brief Gradient 텐서의 ELement를 0으로 초기화하는 메소드
@details Gradient 텐서의 Device 멤버 변수가 CPU인 경우 CPU 메모리에서 초기화하고, CPU인 경우 GPU
메모리에서 초기화한다.
@return Gradient 텐서의 Device 멤버 변수가 Invalid한 값을 가지고 있는 경우 FALSE를 그 외의 경우
TRUE를 반환한다.
*/
template <typename DTYPE>
int LossFunction<DTYPE>::ResetGradient()
{
    if (m_Device == CPU)
    {
        if (m_aGradient)
            m_aGradient->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU)
    {
        if (m_aGradient)
            m_aGradient->Reset(this->GetCudnnHandle());
    }
#endif // if __CUDNN__

    else
        return FALSE;

    return TRUE;
}

#endif // LossFunction_H_
