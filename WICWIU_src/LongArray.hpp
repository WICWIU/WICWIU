#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

/*!
@class LongArray 다차원 Tensor의 데이터를 저장하고 관리하는 클래스.
@details 학습에 사용 될 Tensor의 맴버변수 중 LongArray를 정의하기 위한 클래스.
@details 실질적으로 Tensor클래스의 데이터를 저장하고 관리하기위한 클래스.
@details 데이터를 초기화하고 CPU와 GPU간 데이터의 이동을 가능하게 한다.
*/
template<typename DTYPE> class LongArray {
private:
    DTYPE **m_aaHostLongArray;
    ///< 메모리에 올라가 있는 데이터의 주소 값.

    int m_CapacityOfLongArray;
    ///< LongArray의 총 용량.
    int m_TimeSize;
    ///< Tensor의 TimeSize
    int m_CapacityPerTime;
    ///< Time으로 나누어진 data의 Capacity

    Device m_Device;
    ///< 장치 사용 구분자 (CPU or GPU)
    int m_idOfDevice;
    ///< GPU사용 시, 사용하려는 GPU의 번호.

#ifdef __CUDNN__
    DTYPE **m_aaDevLongArray;
    ///< GPU메모리에 올라가있는 데이터의 주소 값. m_aaHostLongArray와 비슷한 역할을 한다.
#endif  // __CUDNN

private:
    int  Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    int  Alloc(LongArray *pLongArray);
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU(unsigned int idOfDevice);
    void DeleteOnGPU();
    int  MemcpyCPU2GPU();
    int  MemcpyGPU2CPU();
#endif  // __CUDNN

public:
    LongArray(unsigned int pCapacity);
    LongArray(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    LongArray(LongArray *pLongArray);  // Copy Constructor
    virtual ~LongArray();

    int    GetCapacity();
    int    GetTimeSize();
    int    GetCapacityPerTime();
    DTYPE  GetElement(unsigned int index);
    DTYPE& operator[](unsigned int index);
    Device GetDevice();
    int    GetDeviceID();
    DTYPE* GetCPULongArray(unsigned int pTime = 0);

    int    SetDeviceCPU();

    int    Save(unsigned int idxOfParameter);
    int    Load(unsigned int idxOfParameter);
#ifdef __CUDNN__
    int    SetDeviceGPU(unsigned int idOfDevice);

    DTYPE* GetGPUData(unsigned int pTime = 0);

#endif  // if __CUDNN__
};

/*!
@brief LongArray의 맴버 변수들을 초기화하는 메소드
@details pTimeSize을 m_TimeSize, pCapacityPerTime을 m_CapacityPerTime를 초기화 하고,
@details m_CapacityOfLongArray크기의 데이터를 m_CapacityPerTime만큼 m_TimeSize개의 블럭으로 나누어 메모리(RAM)에 할당한다.
@details 할당된 메모리는 0.f로 초기화한다.
@param pTimeSize Alloc할 LongArray의 TimeSize.
@param pCapacity Alloc할 LongArray의 Capacity.
@return 성공 시 TRUE.
*/
template<typename DTYPE> int LongArray<DTYPE>::Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime) {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime)" << '\n';
    #endif  // __DEBUG__

    m_TimeSize        = pTimeSize;
    m_CapacityPerTime = pCapacityPerTime;
    m_aaHostLongArray = new DTYPE *[m_TimeSize];

    for (int i = 0; i < m_TimeSize; i++) {
        m_aaHostLongArray[i] = new DTYPE[m_CapacityPerTime];

        for (int j = 0; j < m_CapacityPerTime; j++) {
            m_aaHostLongArray[i][j] = 0.f;
        }
    }

    m_CapacityOfLongArray = m_TimeSize * m_CapacityPerTime;

    m_Device = CPU;

    return TRUE;
}

/*!
@brief LongArray를 deep copy하는 메소드.
@details pLongArray를 deep copy한다.
@param pLongArray deep copy할 대상 LongArray
@return 성공 시 TRUE
*/
template<typename DTYPE> int LongArray<DTYPE>::Alloc(LongArray *pLongArray) {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::Alloc(LongArray *pLongArray)" << '\n';
    #endif  // __DEBUG__

    m_TimeSize        = pLongArray->GetTimeSize();
    m_CapacityPerTime = pLongArray->GetCapacityPerTime();
    m_aaHostLongArray = new DTYPE *[m_TimeSize];

    for (int i = 0; i < m_TimeSize; i++) {
        m_aaHostLongArray[i] = new DTYPE[m_CapacityPerTime];

        for (int j = 0; j < m_CapacityPerTime; j++) {
            m_aaHostLongArray[i][j] = (*pLongArray)[i * m_CapacityPerTime + j];
        }
    }

    m_CapacityOfLongArray = m_TimeSize * m_CapacityPerTime;

    m_Device = pLongArray->GetDevice();

#ifdef __CUDNN__
    m_idOfDevice = pLongArray->GetDeviceID();

    if (m_Device == GPU) pLongArray->SetDeviceGPU(m_idOfDevice);
#endif  // if __CUDNN__

    return TRUE;
}

/*!
@brief LongArray를 삭제하는 메소드.
@details m_aaHostLongArray가 가리키는 메모리(RAM)들을 free시키고 포인터는 NULL로 초기화한다.
@ref void LongArray<DTYPE>::DeleteOnGPU()
*/
template<typename DTYPE> void LongArray<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aaHostLongArray) {
        for (int i = 0; i < m_TimeSize; i++) {
            if (m_aaHostLongArray[i]) {
                delete[] m_aaHostLongArray[i];
                m_aaHostLongArray[i] = NULL;
            }
        }
        delete[] m_aaHostLongArray;
        m_aaHostLongArray = NULL;
    }

#ifdef __CUDNN__

    this->DeleteOnGPU();
#endif  // __CUDNN__
}

#ifdef __CUDNN__

/*!
@brief LongArray를 GPU메모리에 할당하는 매소드.
@details m_aaHostLongArray와 비슷한 m_aaDevLongArray를 이용하여
@details cudaSetDevice를 통해 GPU를 지정하고 cudaMalloc을 통해 GPU메모리에 LongArray를 할당한다.
@param idOfDevice LongArray를 할당할 GPU번호.
@return 성공 시 TRUE
*/
template<typename DTYPE> int LongArray<DTYPE>::AllocOnGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "LongArray<DTYPE>::AllocOnGPU()" << '\n';
    # endif // __DEBUG__
    m_idOfDevice = idOfDevice;
    checkCudaErrors(cudaSetDevice(idOfDevice));

    if (m_aaDevLongArray == NULL) {
        m_aaDevLongArray = new DTYPE *[m_TimeSize];

        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMalloc((void **)&(m_aaDevLongArray[i]), (m_CapacityPerTime * sizeof(DTYPE))));
        }
    }
    return TRUE;
}

/*!
@brief LongArray를 GPU메모리에서 삭제하는 매소드.
@details  cudaFree를 통해 GPU메모리에 할당 된 m_aaDevLongArray가 가리키는 데모리를 삭제한다.
*/
template<typename DTYPE> void LongArray<DTYPE>::DeleteOnGPU() {
    # if __DEBUG__
    std::cout << "LongArray<DTYPE>::DeleteOnGPU()" << '\n';
    # endif // __DEBUG__

    if (m_aaDevLongArray) {
        for (int i = 0; i < m_TimeSize; i++) {
            if (m_aaDevLongArray[i]) {
                checkCudaErrors(cudaFree(m_aaDevLongArray[i]));
                m_aaDevLongArray[i] = NULL;
            }
        }
        delete[] m_aaDevLongArray;
        m_aaDevLongArray = NULL;
    }
}

/*!
@brief 메모리(RAM)에 있는 LongArray를 GPU메모리로 복사한다.
@details cudaMemcpy를 통해 m_aaHostLongArray가 가리키는 내용을 m_aaDevLongArray로 복사한다.
@return 성공 시 TRUE
*/
template<typename DTYPE> int LongArray<DTYPE>::MemcpyCPU2GPU() {
    # if __DEBUG__
    std::cout << "LongArray<DTYPE>::MemcpyCPU2GPU()" << '\n';
    # endif // __DEBUG__

    if (m_aaDevLongArray != NULL) {
        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aaDevLongArray[i], m_aaHostLongArray[i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        }
    }

    // delete CPU memory
    if (m_aaHostLongArray) {
        for (int i = 0; i < m_TimeSize; i++) {
            if (m_aaHostLongArray[i]) {
                delete[] m_aaHostLongArray[i];
                m_aaHostLongArray[i] = NULL;
            }
        }
        delete[] m_aaHostLongArray;
        m_aaHostLongArray = NULL;
    }
    return TRUE;
}

/*!
@brief GPU메모리에 있는 LongArray를 메모리(RAM)로 복사한다.
@details cudaMemcpy를 통해 m_aaDevLongArray가 가리키는 내용을 m_aaHostLongArray로 복사한다.
@return 성공 시 TRUE
*/
template<typename DTYPE> int LongArray<DTYPE>::MemcpyGPU2CPU() {
    # if __DEBUG__
    std::cout << "LongArray<DTYPE>::MemcpyGPU2CPU()" << '\n';
    # endif // __DEBUG__

    m_aaHostLongArray = new DTYPE *[m_TimeSize];

    for (int i = 0; i < m_TimeSize; i++) {
        m_aaHostLongArray[i] = new DTYPE[m_CapacityPerTime];
    }

    if (m_aaDevLongArray != NULL) {
        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aaHostLongArray[i], m_aaDevLongArray[i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToHost));
        }
    }
    return TRUE;
}

#endif  // if __CUDNN__

/*!
@brief 입력받은 TimeSize와 Capacity크기의 LongArray를 Alloc하는 생성자.
@param pTimeSize Alloc할 LongArray의 TimeSize
@param pCapacity Alloc할 LongArray의 Capacity
@return 없음.
@see LongArray<DTYPE>::Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime)
*/
template<typename DTYPE> LongArray<DTYPE>::LongArray(unsigned int pTimeSize, unsigned int pCapacity) {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::LongArray(unsigned int pTimeSize, unsigned int pCapacity)" << '\n';
    #endif  // __DEBUG__
    m_TimeSize        = 0;
    m_CapacityPerTime = 0;
    m_aaHostLongArray = NULL;
    m_Device          = CPU;
    m_idOfDevice      = -1;
#ifdef __CUDNN__
    m_aaDevLongArray = NULL;
#endif  // __CUDNN
    Alloc(pTimeSize, pCapacity);
}

/*!
@brief LongArray를 deep copy하는 메소드.
@param *pLongArray deep copy할 대상 LongArray
@return 없음.
@see LongArray<DTYPE>::Alloc(LongArray *pLongArray)
*/
template<typename DTYPE> LongArray<DTYPE>::LongArray(LongArray *pLongArray) {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::LongArray(LongArray *pLongArray)" << '\n';
    #endif  // __DEBUG__
    m_TimeSize        = 0;
    m_CapacityPerTime = 0;
    m_aaHostLongArray = NULL;
    m_Device          = CPU;
    m_idOfDevice      = -1;
#ifdef __CUDNN__
    m_aaDevLongArray = NULL;
#endif  // __CUDNN
    Alloc(pLongArray);
}

/*!
@brief LongArray의 소멸자.
@details Delete를 사용하여 해당 LongArray를 메모리에서 삭제한다
@return 없음.
@see void LongArray<DTYPE>::Delete()
*/
template<typename DTYPE> LongArray<DTYPE>::~LongArray() {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::~LongArray()" << '\n';
    #endif  // __DEBUG__
    Delete();
}

template<typename DTYPE> int LongArray<DTYPE>::GetCapacity() {
    return m_TimeSize * m_CapacityPerTime;
}

template<typename DTYPE> int LongArray<DTYPE>::GetTimeSize() {
    return m_TimeSize;
}

template<typename DTYPE> int LongArray<DTYPE>::GetCapacityPerTime() {
    return m_CapacityPerTime;
}

/*!
@brief LongArray의 특정 원소의 값을 반환하는 메소드.
@details 메모리에 있는 LongArray데이터 중 index번째 있는 원소의 값을 반환한다.
@details 단, m_Device가 GPU이면 바로 값을 꺼내올 수 없기 때문에 CPU로 바꿔 준 후 값을 찾아 반환한다.
@return m_aaHostLongArray[index / m_CapacityPerTime][index % m_CapacityPerTime]
@see LongArray<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> DTYPE LongArray<DTYPE>::GetElement(unsigned int index) {
    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! LongArray is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aaHostLongArray[index / m_CapacityPerTime][index % m_CapacityPerTime];
}

/*!
@brief []연산자 Overloading
@details m_aLongArray의 특정 위치에 있는 값을 return할 수 있게 한다.
@details 단, m_Device가 GPU일 시 CPU로 바꿔 준 후 값을 찾아 반환한다.
@details GetElement와 다르게 주소값을 반환하기 때문에 LongArray의 값을 변경 할 수 있다.
@see LongArray<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> DTYPE& LongArray<DTYPE>::operator[](unsigned int index) {
    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! LongArray is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aaHostLongArray[index / m_CapacityPerTime][index % m_CapacityPerTime];
}

template<typename DTYPE> Device LongArray<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int LongArray<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

/*!
@brief m_aaHostLongArray중 pTime에 있는 LongArray를 반환하는 메소드.
@details m_CapacityOfLongArray를 m_TimeSize로 나눈 LongArray블럭 중 pTime번째의 LongArray블럭을 반환한다.
@details 단, m_Device가 GPU일 시 CPU로 바꿔 준 후 값을 찾아 반환한다.
@return m_aaHostLongArray[pTime]
@see LongArray<DTYPE>::SetDeviceCPU()
*/
template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetCPULongArray(unsigned int pTime) {
    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! LongArray is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aaHostLongArray[pTime];
}

/*!
@brief LongArray의 m_Device를 CPU로 바꾸는 메소드.
@details m_Device를 CPU로 바꾼다. CUDNN이 있을 경우 GPU 메모리의 값들을 CPU메모리로 복사한다.
@return 없음.
@see MemcpyGPU2CPU()
*/
template<typename DTYPE> int LongArray<DTYPE>::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "LongArray<DTYPE>::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;
#ifdef __CUDNN__
    this->MemcpyGPU2CPU();
#endif  // __CUDNN__
    return TRUE;
}

/*!
@brief LongArray의 데이터를 파일에 저장하는 메소드.
@details fwrite함수를 통해 *fileForSave가 가리키는 파일에 LongArray데이터를 쓴다.
@details 단, m_Device가 GPU일 시 CPU로 바꿔 준 후 값을 찾아 반환한다.
@param *fileForSave 데이터를 저장할 file을 가리는 포인터.
@return 성공 시 TRUE.
@see LongArray<DTYPE>::SetDeviceCPU()
*/

template<typename DTYPE> int LongArray<DTYPE>::Save(unsigned int idxOfParameter) {
    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! LongArray is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    #ifdef __BINARY__
    std::cout << "save" << '\n';
    #endif  // __BINARY__

    char filename[idxOfParameter];
    sprintf(filename, "%d", idxOfParameter);
    FILE *fp = fopen(filename, "wb");

    if (!fwrite(&m_CapacityPerTime, sizeof(int), 1, fp)) {
        printf("Failed to write Data from binary file in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(-1);
    }

    for (int i = 0; i < m_TimeSize; i++) {
        if (!fwrite(m_aaHostLongArray[i], sizeof(DTYPE), m_CapacityPerTime, fp)) {
            printf("Failed to write Data from binary file in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            exit(-1);
        }
    }
    fclose(fp);

    return TRUE;
}

template<typename DTYPE> int LongArray<DTYPE>::Load(unsigned int idxOfParameter) {
    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! LongArray is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    #ifdef __BINARY__
    std::cout << "load" << '\n';
    #endif  // __BINARY__

    int capacityOfData = idxOfParameter;
    char filename[idxOfParameter];
    sprintf(filename, "%d", idxOfParameter);

    FILE *fp = fopen(filename, "rb");

    printf("%s\n", filename);
    int filesize = ftell(fp);

    if (!fread(&capacityOfData, sizeof(int), 1, fp)) {
        printf("Failed to read Data from binary file in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(-1);
    }

    if (capacityOfData != 0) {
        if (capacityOfData <= m_CapacityPerTime) {
            for (int i = 0; i < m_TimeSize; i++) {
                if (!fread(m_aaHostLongArray[i], sizeof(DTYPE), capacityOfData, fp)) {
                    printf("Failed to read Data from binary file in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                    exit(-1);
                }
            }
        }
    }

    fclose(fp);

    return TRUE;
}

#ifdef __CUDNN__
/*!
@brief LongArray의 m_Device를 GPU로 바꿔주는 메소드.
@details LongArray의 m_Device를 GPU로 바꾼다.
@details m_aaDevLongArray가 NULL포인터 일 경우 AllocOnGPU를 통해 idOfDevice번째 GPU에 LongArray를 할당한다.
@details idOfDevice와 m_idOfDevice가 같지 않을 경우 현재 할당된 GPU에서 LongArray를 삭제한 후 idOfDevice번째 GPU에 새로 할당한다.
@param idOfDevice 할당 할 GPU번호.
@return 성공 시 TRUE.
@see LongArray<DTYPE>::AllocOnGPU, LongArray<DTYPE>::DelteOnGPU, LongArray<DTYPE>::MemcpyCPU2GPU()
*/
template<typename DTYPE> int LongArray<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "LongArray<DTYPE>::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__

    m_Device = GPU;

    if (m_aaDevLongArray == NULL) this->AllocOnGPU(idOfDevice);

    if (idOfDevice != m_idOfDevice) {
        this->DeleteOnGPU();
        this->AllocOnGPU(idOfDevice);
    }
    this->MemcpyCPU2GPU();

    return TRUE;
}

/*!
@brief m_aaDevLongArray중 pTime에 있는 LongArray를 반환하는 메소드.
@details m_TimeSize개의 time Dimension중 pTime번째의 LongArray를 반환한다.
@details 단, m_Device가 CPU일 시 GPU로 바꿔 준 후 값을 찾아 반환한다.
@return m_aaDevLongArray[pTime]
@see LongArray<DTYPE>::SetDeviceGPU()
*/
template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetGPUData(unsigned int pTime) {
# if __DEBUG__

    if (m_Device == CPU) {
        printf("Warning! LongArray is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

# else // if __DEBUG__

#  if __ACCURATE__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }
#  endif // __ACCURATE__

# endif // __DEBUG__

    return m_aaDevLongArray[pTime];
}

#endif  // if __CUDNN__

#endif  // __DATA__
