#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.hpp"
#include "Container.hpp"

/*!
 * @brief Operator의 현재 상태를 나타내는  enum class
 * @details TRAINING:학습 중, ACCUMULATING:, INFERENCING:accuracy를 구하는 중
 */
enum Mode {
    TRAIN,
    ACCUMULATE,
    INFERENCE,
};

/*!
 * @class Operator class
 * @details 본 프래임워크의 가장 작은 연산 단위.
 */
template<typename DTYPE> class Operator {
private:
    Container<Operator<DTYPE> *> *m_apOutput;
    ///< Operator의 m_aaResult값을 사용할 Operator들의 주소 값.
    Container<Operator<DTYPE> *> *m_apInput;
    ///< Operator에 input으로  들어오는 Operator들의 주소 값.
    Container<Tensor<DTYPE> *> *m_aaResult;
    ///< Operator의 결과 값.
    Container<Tensor<DTYPE> *> *m_aaGradient;
    ///< Operator의 Gradiuent값들의 Array.
    std::string m_name;
    ///< Operator에 사용자가 부여한 이름.
    Device m_Device;
    ///< Operator가 사용하고 있는 Device, 해당 Device의 메모리에 Operator가 있다.
    int m_idOfDevice;
    ///< m_Device가 GPU일 경우 사용하는 GPU번호.
    Mode m_Mode;
    ///< Operator의 Mode.
    int m_isParameter;
    ///< Operator가 파라미터인지 알려주는 값.
    int m_isTrainable;
    ///< Operator가 학습가능한 Operator인지 알려주는 값.
    int m_Loadflag;

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
    ///< cudnn 라이브러리를 가리키는 포인터.
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
    Operator(std::string pName = "NO NAME", int pLoadflag = TRUE);
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME", int pLoadflag = TRUE);
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME", int pLoadflag = TRUE);
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, std::string pName = "NO NAME", int pLoadflag = TRUE);
    Operator(int numInput, ...);
    virtual ~Operator();

    int                                   AddEdgebetweenOperators(Operator<DTYPE> *pInput);
    int                                   AddEdgebetweenOperators(int numInput, va_list ap);
    int                                   AddEdgebetweenOperators(int numInput, ...);
    int                                   AddResult(Tensor<DTYPE> *pTensor);
    int                                   AddGradient(Tensor<DTYPE> *pTensor);
    int                                   AddDelta(Tensor<DTYPE> *pTensor);
    int                                   SetResult(Tensor<DTYPE> *pTensor);     // 0 or 1 일 때만 진행 가능
    int                                   SetGradient(Tensor<DTYPE> *pTensor);
    int                                   SetDelta(Tensor<DTYPE> *pTensor);

    int                                   SetDevice(Device pDevice);
    int                                   SetDeviceID(unsigned int idOfDevice);

    int                           SetIsTensorholder(int pIsParameter);
    int                           SetIsTrainable(int pIsTrainable);

    virtual int                           SetModeTrain();
    virtual int                           SetModeAccumulate();
    virtual int                           SetModeInference();

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

    // virtual void                          PrintInformation();
    virtual void                          PrintInformation(int level);

    virtual void                          SetDeviceCPU();

    virtual int                           SetResultOnCPU();
    virtual int                           SetGradientOnCPU();

    virtual int                           Save(char *nameOfFile);
    virtual int                           Load(char *nameOfFile);

    virtual int                           Load(FILE *fp);
    virtual int                           Save(FILE *fp);
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

//////////////////////////////////////////////////////////////////////////////// for private method

/*!
 * @brief Operator의 맴버 변수들중 포인터들을 메모리에 할당하는 매소드.
 * @details 맴버변수 m_apOutput, m_apInput, m_aaResult, m_aaGradient들을 메모리에 할당한다
 * @return 성공시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::Alloc() {
    m_apOutput   = new Container<Operator<DTYPE> *>();
    m_apInput    = new Container<Operator<DTYPE> *>();
    m_aaResult   = new Container<Tensor<DTYPE> *>();
    m_aaGradient = new Container<Tensor<DTYPE> *>();

    return TRUE;
}

/*!
 * @brief Operator와 다수의 다른 Operator들을 연결시키는 매소드.
 * @details AddEdgebetweenOperators매소드를 통해 파라미터로 전달받은 Operator의 주소값을 이용해을 연결시킨다.
 * @param numInput 연결 할 Operator의 갯수.
 * @param ... Operator와 연결할 input Operator들.
 * @return 성공 시 TRUE, 실패 시 FALSE.
 * @ref Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput)
 */
template<typename DTYPE> int Operator<DTYPE>::Alloc(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);

        if (!temp) {
            null_count++;
        } else {
            this->AddEdgebetweenOperators(temp);
        }
    }

    va_end(ap);

    if (null_count) {
        numInput = numInput - null_count;

        for (int i = 0; i < numInput; i++) {
            delete (*m_apInput)[i];
        }
        delete m_apInput;
        m_apInput = NULL;

        printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

/*!
 * @brief Operator를 메모리에 삭제하는 매소드.
 * @details 메모리에 할당했던 변수들을 삭제하고 포인터들을 NULL로 초기화한다.
 */
template<typename DTYPE> void Operator<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    int size = 0;

    if (m_aaResult) {
        size = m_aaResult->GetSize();
        Tensor<DTYPE> **ResultContainer = m_aaResult->GetRawData();

        for (int i = 0; i < size; i++) {
            delete ResultContainer[i];
            ResultContainer[i] = NULL;
        }

        delete m_aaResult;
        m_aaResult = NULL;
    }

    if (m_aaGradient) {
        size = m_aaGradient->GetSize();
        Tensor<DTYPE> **GradientContainer = m_aaGradient->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaGradient)[i]) {
                delete GradientContainer[i];
                GradientContainer[i] = NULL;
            }
        }

        delete m_aaGradient;
        m_aaGradient = NULL;
    }

    if (m_apOutput) {
        delete m_apOutput;
        m_apOutput = NULL;
    }

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }
}

// Add Graph Edge

/*!
 * @brief Operator의 m_apInput을 설정하는 함수.
 * @details 다른 Operator의 주소 값을 받아 Operator의 input값(m_apInput)으로 설정한다.
 * @param pInput input으로 설정 할 Operator들의 주소 값.
 * @return 성공 시 TRUE, 실패 시 FALSE
 * @ref int Push(DTYPE pElement)
 */
template<typename DTYPE> int Operator<DTYPE>::AddInputEdge(Operator<DTYPE> *pInput) {
    try {
        m_apInput->Push(pInput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

/*!
 * @brief Operator의 m_apOutput을 설정하는 함수.
 * @details 다른 Operator의 주소 값을 받아 Operator의 output값(m_apOutput)으로 설정한다.
 * @param pOutput Operator의 output을 input으로 사용할 Operator들의 주소 값.
 * @return 성공 시 TRUE, 실패 시 FALSE
 * @ref int Push(DTYPE pElement)
 */
template<typename DTYPE> int Operator<DTYPE>::AddOutputEdge(Operator<DTYPE> *pOutput) {
    try {
        m_apOutput->Push(pOutput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
 * @brief Operator의 생성자.
 * @details 파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한다.
 * @param pName 사용자가 설정 할 Operator의 이름.
 */
template<typename DTYPE> Operator<DTYPE>::Operator(std::string pName, int pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    m_idOfDevice  = -1;
    m_Loadflag    = TRUE;
    Alloc();
}

/*!
 * @brief Operator의 생성자.
 * @details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
 * @param pInput Operator와 연결 할 Operator들의 주소 값들.
 * @param pName 사용자가 설정 할 Operator의 이름.
 * @ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
 */
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput, std::string pName, int pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    m_idOfDevice  = -1;
    m_Loadflag    = TRUE;
    Alloc();
    AddEdgebetweenOperators(1, pInput, pLoadflag);
}

/*!
 * @brief Operator의 생성자.
 * @details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
 * @param pInput0 Operator와 연결 할 Operator들의 주소 값들.
 * @param pInput1 Operator와 연결 할 Operator들의 주소 값들.
 * @param pName 사용자가 설정 할 Operator의 이름.
 * @ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
 */
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName, int pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    m_idOfDevice  = -1;
    m_Loadflag    = TRUE;
    Alloc();
    AddEdgebetweenOperators(2, pInput0, pInput1, pLoadflag);
}

/*!
 * @brief Operator의 생성자.
 * @details파라미터로 전달 받은 pName을 m_name으로 설정 하고 나머지는 변수들은 NULL, CPU, TRAINING으로 초기화 한 뒤, AddEdgebetweenOperators를 통해 Operator들을 서로 연결한다.
 * @param pInput0 Operator와 연결 할 Operator들의 주소 값들.
 * @param pInput1 Operator와 연결 할 Operator들의 주소 값들.
 * @param pInput2 Operator와 연결 할 Operator들의 주소 값둘.
 * @param pName 사용자가 설정 할 Operator의 이름.
 * @ref Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...)
 */
template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, std::string pName, int pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    m_idOfDevice  = -1;
    m_Loadflag    = TRUE;
    Alloc();
    AddEdgebetweenOperators(3, pInput0, pInput1, pInput2, pLoadflag);
}

template<typename DTYPE> Operator<DTYPE>::Operator(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = "";
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    m_idOfDevice  = -1;
    Alloc();

    va_list ap;
    va_start(ap, numInput);
    AddEdgebetweenOperators(numInput, ap);
    va_end(ap);
}

/*!
 * @brief Operator의 소멸자
 * @details Delete 매소드를 이용해 삭제한다.
 */
template<typename DTYPE> Operator<DTYPE>::~Operator() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::~Operator()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

/*!
 * @brief Operator와 다른 Operator들을 서로 연결한다.
 * @details AddInputEdge, AddOutputEdge 매소드를 이옹해 Operator와 다른 Operator 연결한다.
 * @param pInput 연결 할 다른 Operator의 주소 값.
 * @ref int Operator<DTYPE>::AddInputEdge(Operator<DTYPE> *pInput), int Operator<DTYPE>::AddOutputEdge(Operator<DTYPE> *pOutput)
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    this->AddInputEdge(pInput);
    pInput->AddOutputEdge(this);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(int numInput, va_list ap) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;


    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);

        if (!temp) {
            null_count++;
        } else {
            this->AddEdgebetweenOperators(temp);
        }
    }

    if (null_count) {
        numInput = numInput - null_count;

        for (int i = 0; i < numInput; i++) {
            delete (*m_apInput)[i];
        }
        delete m_apInput;
        m_apInput = NULL;

        printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

/*!
 * @brief Operator와 다른 Operator들을 서로 연결한다.
 * @details AddEdgebetweenOperators 매소드를 이옹해 Operator와 다수의 다른 Operator들을 연결한다.
 * @param numInput 연결 할 input들의 갯수.
 * @param ... 연결할 다수의 다른 Operator.
 * @ref int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput)
 * @return 성공 시 TRUE, 실패 시 FALSE.
 */
template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);

        if (!temp) {
            null_count++;
        } else {
            this->AddEdgebetweenOperators(temp);
        }
    }

    va_end(ap);

    if (null_count) {
        numInput = numInput - null_count;

        for (int i = 0; i < numInput; i++) {
            delete (*m_apInput)[i];
        }
        delete m_apInput;
        m_apInput = NULL;

        printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    if (m_aaResult->GetSize()) {
        Tensor<DTYPE> *temp = m_aaResult->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaResult->Push(pTensor);
    return TRUE;
}

/*!
 * @brief 파라미터로 받은 Tensor를 결과 값으로 설정한다.
 * @details 파라미터로 받은 pTensor를 m_aaResult애 저장한다.
 * @param pTensor m_aaResult에 저장 할 Tensor.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::AddResult(Tensor<DTYPE> *pTensor) {
    m_aaResult->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

/*!
 * @brief 파라미터로 받은 Tensor를 gradient값으로 설정한다.
 * @details 파라미터로 받은 pTensor를 m_aaGradient에 저장한다.
 * @param pTensor m_aaGradient에 저장 할 Tensor.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::AddGradient(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDelta(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

/*!
 * @brief 파라미터로 받은 Tensor를 Delta값으로 설정한다.
 * @details 파라미터로 받은 pTensor를 m_aaGradient에 저장한다.
 * @param pTensor m_aaGradient에 저장 할 Tensor.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::AddDelta(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDevice(Device pDevice) {
    m_Device = pDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTensorholder(int pIsParameter) {
    m_isParameter = pIsParameter;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTrainable(int pIsTrainable) {
    m_isTrainable = pIsTrainable;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeTrain() {
    m_Mode = TRAIN;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeAccumulate() {
    m_Mode = ACCUMULATE;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeInference() {
    m_Mode = INFERENCE;
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetOutput() {
    return m_apOutput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetOutputContainer() {
    return m_apOutput;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetInput() {
    return m_apInput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetInputContainer() {
    return m_apInput;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetResult() const {
    return (*m_aaResult)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetResultContainer() {
    return m_aaResult;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetGradient() const {
    return (*m_aaGradient)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetGradientContainer() {
    return m_aaGradient;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetDelta() const {
    return (*m_aaGradient)[0];
    // return (*m_aaDelta)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetDeltaContainer() {
    return m_aaGradient;
    // return m_aaDelta;
}

template<typename DTYPE> std::string Operator<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Device Operator<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int Operator<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTensorholder() {
    return m_isParameter;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTrainable() {
    return m_isTrainable;
}

/*!
 * @brief  ForwardPropagate 매소드. 실제 구현은 파생 클래스에서 정의된다.
 * @param pTime ForwardPropagate할 데이터의 Time값.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__

    #endif  // __DEBUG__
    return TRUE;
}

/*!
 * @brief  BackwardPropagate 매소드. 실제 구현은 파생 클래스에서 정의된다.
 * @param pTime forwardPropagate했던 데이터의 Time값.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__

    #endif  // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetResult() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetGradient() {
    int size = m_aaGradient->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

// template<typename DTYPE> void Operator<DTYPE>::PrintInformation() {
// std::cout << this->GetName() << " : ";
// std::cout << this->GetResult()->GetShape() << '\n';
// }

/*!
 * @brief Operator정보(이름과 Shape)를 출력한다.
 */
template<typename DTYPE> void Operator<DTYPE>::PrintInformation(int level) {
    for (int j = 0; j < level; j++) {
        std::cout << "-- ";
    }
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';
}

template<typename DTYPE> void Operator<DTYPE>::SetDeviceCPU() {
    if (m_Device != CPU) {
        this->SetDevice(CPU);
        this->SetResultOnCPU();
        this->SetGradientOnCPU();
    }
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnCPU() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnCPU() {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Save(char *nameOfFile) {
    FILE *fp = fopen(nameOfFile, "wb");

    this->Save(fp);

    fclose(fp);

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Load(char *nameOfFile) {
    FILE *fp = fopen(nameOfFile, "rb");

    this->Load(fp);

    fclose(fp);

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Save(FILE *fp) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Save(fp);
    }
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Load(FILE *fp) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Load(fp);
    }
    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> int Operator<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnGPU(unsigned int idOfDevice) {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

// template<typename DTYPE> void Operator<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// this->SetDevice(GPU);
// this->SetDeviceID(idOfDevice);
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

/*!
 * @brief Operator가 GPU에서 연산 될 수 있도록 하는 매소드.
 * @details Operator의 정보들을 지정된 GPU의 메모리로 복사한다.
 * @param pCudnnHandle cudnn 라이브러리를 가리키는 포인터.
 * @param idOfDevice 사용 할 GPU 번호
 */
template<typename DTYPE> void Operator<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    if (m_Device != GPU) {
        checkCudaErrors(cudaSetDevice(idOfDevice));
        this->SetCudnnHandle(pCudnnHandle);
        this->SetDevice(GPU);
        this->SetDeviceID(idOfDevice);
        this->SetResultOnGPU(idOfDevice);
        this->SetGradientOnGPU(idOfDevice);
        this->InitializeAttributeForGPU(idOfDevice);
    }
}

template<typename DTYPE> cudnnHandle_t& Operator<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

/*!
 * @brief  ForwardPropagateOnGPU 매소드. 실제 구현은 파생 클래스에서 정의된다.
 * @param pTime ForwardPropagate할 데이터의 Time값.
 * @return 성공 시 TRUE.
 */
template<typename DTYPE> int Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::ForwardPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::BackPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

#endif  // __CUDNN__

#endif  // OPERATOR_H_
