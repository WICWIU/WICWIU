#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

template<typename DTYPE> class LongArray {
private:
    DTYPE **m_aaHostLongArray;

    int m_CapacityOfLongArray;
    int m_TimeSize;
    int m_CapacityPerTime;

    Device m_Device;
    int m_idOfDevice;

#ifdef __CUDNN__
    DTYPE **m_aaDevLongArray;
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

    std::cout << "idx" << '\n';
    std::cout << idxOfParameter << '\n';
    std::cout << "capacityofdata" << '\n';
    std::cout << capacityOfData << '\n';
    std::cout << "filename" << '\n';
    printf("%s\n", filename);
    int filesize = ftell(fp);
    std::cout << "filesize" << '\n';
    std::cout << filesize << '\n';

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
