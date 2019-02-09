#include "Shape.hpp"


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief Shape 클래스를 동적 할당하는 메소드
@details Shape의 Rank와 각 축의 Dimension을 매개변수로 받아 Shape 클래스를 동적으로 할당한다.
@param pRank 할당하고자 하는 Shape 클래스의 Rank
@param ... 할당하고자 하는 Shape 클래스의 각 축의 Dimension 리스트
@return 할당 성공 시 TRUE, 할당 실패 시 FALSE
*/
int Shape::Alloc(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = va_arg(ap, int);
    }
    va_end(ap);

    m_Device = CPU;

    return TRUE;
}

/*!
@brief Shape 클래스를 깊은 복사(Deep Copy)하여 동적으로 할당하는 메소드
@param pShape 깊은 복사(Deep Copy)하고자 하는 Shape 클래스
@return 할당 성공 시 TRUE, 할당 실패 시 FALSE
*/
int Shape::Alloc(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Alloc(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    try {
        m_Rank = pShape->GetRank();
        m_aDim = new int[m_Rank];

        for (int i = 0; i < m_Rank; i++) {
            m_aDim[i] = (*pShape)[i];
        }
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    m_Device = pShape->GetDevice();

#ifdef __CUDNN__
    m_idOfDevice = pShape->GetDeviceID();

    if (m_Device == GPU) SetDeviceGPU(m_idOfDevice);
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief Shape 클래스를 생성하기 위해 동적으로 할당한 메모리 공간을 반환하는 메소드
@details Shape 클래스 동적 할당에 사용했던 메모리 공간을 반환하고 해당 메모리 공간을 NULL로 초기화한다.
@return 없음
*/
void Shape::Delete() {
    #ifdef __DEBUG__
    std::cout << "Shape::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aDim) {
        delete[] m_aDim;
        m_aDim = NULL;
    }

#ifdef __CUDNN__
    DeleteOnGPU();
#endif  // if __CUDNN__
}

#ifdef __CUDNN__
/*!
@brief GPU을 사용하기 위해 GPU 메모리에 해당 Shape 클래스의 정보를 동적으로 할당하는 메소드
@details cuda와 cudnn 라이브러리를 이용하여, 매개변수로 받은 GPU 번호에 해당되는 GPU의 메모리 공간에 Descriptor 변수를 할당한다
@details GPU내에서 Tensor의 형식은 batch, channel, row, colunm순서로 배치되도록 지정한다.
@param idOfDevice Descriptor를 할당하고자 하는 GPU의 번호
@return 성공 시 TRUE
*/
int Shape::AllocOnGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::AllocOnGPU()" << '\n';
    # endif // __DEBUG__

    m_idOfDevice = idOfDevice;
    checkCudaErrors(cudaSetDevice(idOfDevice));

    if (m_desc == NULL) {
        if (m_Rank == 5) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[1], m_aDim[2], m_aDim[3], m_aDim[4]));
        } else if (m_Rank == 4) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[0], m_aDim[1], m_aDim[2], m_aDim[3]));
        } else ;
    } else ;

    return TRUE;
}

/*!
@brief Descriptor를 동적으로 할당했던 GPU 메모리 공간을 반환하는 메소드
@details cudnn 라이브러리를 사용하여, Descriptor 동적 할당에 사용했던 메모리 공간을 반환하고 해당 메모리 공간을 NULL로 초기화한다.
@return 없음
*/
void Shape::DeleteOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::DeleteOnGPU()" << '\n';
    # endif // __DEBUG__

    if (m_desc) {
        checkCUDNN(cudnnDestroyTensorDescriptor(m_desc));
        m_desc = NULL;
    }
}

/*!
@brief GPU 메모리내에 할당 된 데이터를 해제하고 새로 할당한다.
@details 사전에 정의 한 DeleteOnGPUdhk AllocOnGPU 메소드를 이용하여, 기존에  GPU에 할당 되어있던 Descriptor를 할당 해제 시키, 새로운 Descriptor를 할당 시킨다.
@return 성공 시 TRUE.
*/
int Shape::ReShapeOnGPU() {
    # if __DEBUG__
    std::cout << "Shape::ReShapeOnGPU()" << '\n';
    # endif // __DEBUG__

    DeleteOnGPU();

    if (m_idOfDevice == -1) {
        std::cout << "you need to set device GPU first before : ReShapeOnGPU" << '\n';
        exit(-1);
    }
    AllocOnGPU(m_idOfDevice);

    return TRUE;
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
@brief 5D-Shape 생성자
@details 5개의 축의 Dimension을 매개변수로 받아 Shape 클래스를 생성하는 생성자
@param pSize0 첫 번째 축의 Dimension 크기
@param pSize1 두 번째 축의 Dimension 크기
@param pSize2 세 번째 축의 Dimension 크기
@param pSize3 네 번째 축의 Dimension 크기
@param pSize4 다섯 번째 축의 Dimension 크기
@return 없음
@see Shape::Alloc(int pRank, ...)
*/
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief 4D-Shape 생성자
@details 4개의 축의 Dimension을 매개변수로 받아 Shape 클래스를 생성하는 생성자
@param pSize0 첫 번째 축의 Dimension 크기
@param pSize1 두 번째 축의 Dimension 크기
@param pSize2 세 번째 축의 Dimension 크기
@param pSize3 네 번째 축의 Dimension 크기
@see Shape::Alloc(int pRank, ...)
@return 없음
*/
Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(4, pSize0, pSize1, pSize2, pSize3);
}

/*!
@brief 3D-Shape 생성자
@details 3개의 축의 Dimension을 매개변수로 받아 Shape 클래스를 생성하는 생성자
@param pSize0 첫 번째 축의 Dimension 크기
@param pSize1 두 번째 축의 Dimension 크기
@param pSize2 세 번째 축의 Dimension 크기
@see Shape::Alloc(int pRank, ...)
@return 없음
*/
Shape::Shape(int pSize0, int pSize1, int pSize2) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(3, pSize0, pSize1, pSize2);
}

/*!
@brief 2D-Shape 생성자
@details 2개의 축의 Dimension을 매개변수로 받아 Shape 클래스를 생성하는 생성자
@param pSize0 첫 번째 축의 Dimension 크기
@param pSize1 두 번째 축의 Dimension 크기
@return 없음
@see Shape::Alloc(int pRank, ...)
*/
Shape::Shape(int pSize0, int pSize1) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(2, pSize0, pSize1);
}

/*!
@brief 1D-Shape 생성자
@details 1개의 축의 Dimension을 매개변수로 받아 Shape 클래스를 생성하는 생성자
@param pSize0 첫 번째 축의 Dimension 크기
@return 없음
@see Shape::Alloc(int pRank, ...)
*/
Shape::Shape(int pSize0) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(1, pSize0);
}

/*!
@brief Shape 클래스를 매개변수로 받아 깊은 복사(Deep Copy)하는 Shape 생성자
@param pShape 깊은 복사(Deep Copy)의 대상이 되는 Shape 클래스
@return 없음
@see Shape::Alloc(Shape *pShape)
*/
Shape::Shape(Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "Shape::Shape(Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    m_Rank       = 0;
    m_aDim       = NULL;
    m_idOfDevice = -1;
#ifdef __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__

    Alloc(pShape);
}

/*!
@brief Shape 클래스 소멸자
@details 해당 Shape 클래스를 위해 동적으로 할당된 메모리 공간을 반환하고 클래스를 소멸한다.
@return 없음
@see Shape::Delete()
*/
Shape::~Shape() {
    #ifdef __DEBUG__
    std::cout << "Shape::~Shape()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

/*!
@brief Shape 클래스의 Rank 멤버 변수를 반환하는 메소드
@return m_Rank
*/
int Shape::GetRank() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_Rank;
}

/*!
@brief Rank 인덱스를 파라미터로 받아 Dimension을 반환하는 메소드
@param pRanknum Dimension을 반환하고자 하는 축의 번호
@return 성공 시 m_aDim[pRanknum], 실패 시 예외 처리
*/
int Shape::GetDim(int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief Shape 클래스의 Device 멤버 변수를 반환하는 메소드
@return m_Device
*/
Device Shape::GetDevice() {
    #ifdef __DEBUG__
    std::cout << "Shape::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

/*!
@brief Shape 클래스의 idOfDevice 멤버 변수를 반환하는 메소드
@return m_idOfDevice
*/
int Shape::GetDeviceID() {
    return m_idOfDevice;
}

/*!
@brief []연산자 오버로딩
@details Rank 인덱스를 파라미터로 받아 Dimension을 반환하는 메소드
@param pRanknum Dimension을 반환하고자 하는 축의 번호
@return 성공 시 m_aDim[pRanknum], 실패 시 예외 처리
*/
int& Shape::operator[](int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Shape::operator[](int pRanknum)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/*!
@brief 새로운 Shape을 만들어 반환 하는 메소드.
@details 파마미터로 전달받은 각 축의  Dimension정보를 바탕으로 새로운 Shape을 생성하여 반환한다.
@param pSize0 Time의 Dimension
@param pSize1 Batch의 Dimension
@param pSize2 Channel의 Dimension
@param pSize3 Row의 Dimension
@param pSize4 Column의 Dimension
@return 파라미터 정보를 바탕으로 만든 Shape
*/
int Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    return ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

/*!
@brief Shape 각 축의 Dimension정보를 초기화 하는 메소드.
@details 파마미터로 전달받은 각 축의  Dimension정보를 m_aDim에 저장한다.
@param pRank Shape을 이루는 축의 갯수를 나타내는 변수.
@param ... 각 축의 Dimension정보를 가지고 있는 가변인자.
@return 성공 시 TRUE
*/
int Shape::ReShape(int pRank, ...) {
    #ifdef __DEBUG__
    std::cout << "Shape::ReShape(int pRank, ...)" << '\n';
    #endif  // __DEBUG__

    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        if (m_aDim) {
            delete[] m_aDim;
            m_aDim = NULL;
        }
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    for (int i = 0; i < pRank; i++) {
        m_aDim[i] = va_arg(ap, int);
    }

    va_end(ap);

#ifdef __CUDNN__

    if (m_Device == GPU) ReShapeOnGPU();
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief Shape 클래스의 Device 멤버 변수를 CPU로 변경한다.
@return TRUE
*/
int Shape::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "Shape::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;

    return TRUE;
}

#ifdef __CUDNN__

/*!
@brief Shape 클래스의 Device 멤버 변수를 GPU로 변경한다.
@details CPU에서 GPU로 전환 시, 매개변수로 받은 번호에 해당하는 GPU의 메모리에 필요한 공간을 동적으로 할당한다.
@param 사용하고자 하는 GPU의 번호
@return TRUE
@see Shape::AllocOnGPU(unsigned int idOfDevice)
*/
int Shape::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Shape::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__

    m_Device = GPU;

    if (m_desc == NULL) AllocOnGPU(idOfDevice);

    return TRUE;
}

/*!
@brief GPU내 Descriptor를 반ㅇ환하는 함수.
@details cudnn의 Tensor Descriptor, GPU 연산에서 사용하는 Tensor의 shape 정보를 담고 있는 포인터 변수를 반환한다.
@return GPU내 Tensor의 shape정보를 담고 있는 포인터 변수
*/
cudnnTensorDescriptor_t& Shape::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Shape::GetDescriptor()" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_desc;
}

#endif  // __CUDNN__


std::ostream& operator<<(std::ostream& pOS, Shape *pShape) {
    #ifdef __DEBUG__
    std::cout << "std::ostream& operator<<(std::ostream& pOS, Shape *pShape)" << '\n';
    #endif  // __DEBUG__

    int rank = pShape->GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << (*pShape)[i] << ", ";
    pOS << "]";
    return pOS;
}
