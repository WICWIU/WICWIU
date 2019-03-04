#ifndef TENSOR_H_
#define TENSOR_H_

#include "Shape.hpp"
#include "LongArray.hpp"

enum IsUseTime {
    UseTime,
    NoUseTime
};

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    LongArray<DTYPE> *m_aLongArray;
    Device m_Device;
    int m_idOfDevice;
    IsUseTime m_IsUseTime;

private:
    int  Alloc(Shape *pShape, IsUseTime pAnswer);
    int  Alloc(Tensor *pTensor);
    void Delete();

public:
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);  // For 5D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer = UseTime);  // For 4D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer = UseTime);  // For 3D-Tensor
    Tensor(int pSize0, int pSize1, IsUseTime pAnswer = UseTime);  // For 2D-Tensor
    Tensor(int pSize0, IsUseTime pAnswer = UseTime);  // For 1D-Tensor
    Tensor(Shape *pShape, IsUseTime pAnswer = UseTime);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    Shape                  * GetShape();
    int                      GetRank();
    int                      GetDim(int pRanknum);
    LongArray<DTYPE>       * GetLongArray();
    int                      GetCapacity();
    int                      GetElement(unsigned int index);
    DTYPE                  & operator[](unsigned int index);
    Device                   GetDevice();
    IsUseTime                GetIsUseTime();
    DTYPE                  * GetCPULongArray(unsigned int pTime = 0);

    int                      GetTimeSize(); // 추후 LongArray의 Timesize 반환
    int                      GetBatchSize(); // 삭제 예정
    int                      GetChannelSize(); // 삭제 예정
    int                      GetRowSize(); // 삭제 예정
    int                      GetColSize(); // 삭제 예정


    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3);
    int                      ReShape(int pSize0, int pSize1, int pSize2);
    int                      ReShape(int pSize0, int pSize1);
    int                      ReShape(int pSize0);

    void                     Reset();


    void                     SetDeviceCPU();

    int                      Save(unsigned int idxOfParameter);
    int                      Load(unsigned int idxOfParameter);
#ifdef __CUDNN__
    void                     SetDeviceGPU(unsigned int idOfDevice);

    DTYPE                  * GetGPUData(unsigned int pTime = 0);
    cudnnTensorDescriptor_t& GetDescriptor();

    void                     Reset(cudnnHandle_t& pCudnnHandle);


#endif  // if __CUDNN__


    static Tensor<DTYPE>* Random_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Random_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(Shape *pShape, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer = UseTime);
};

//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape    = pShape;
        m_IsUseTime = pAnswer;

        int rank = pShape->GetRank();

        int pTime            = 1;
        int pCapacityPerTime = 1;

        if (m_IsUseTime == UseTime) {
            pTime = (*pShape)[0];

            for (int i = 1; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else if (m_IsUseTime == NoUseTime) {
            for (int i = 0; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else return FALSE;

        m_aLongArray = new LongArray<DTYPE>(pTime, pCapacityPerTime);
    }

    m_Device = CPU;

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor)" << '\n';
    #endif  // __DEBUG__

    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape     = new Shape(pTensor->GetShape());
        m_aLongArray = new LongArray<DTYPE>(pTensor->GetLongArray());
        m_Device     = pTensor->GetDevice();
        m_IsUseTime  = pTensor->GetIsUseTime();
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aLongArray) {
        delete m_aLongArray;
        m_aLongArray = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(new Shape(pSize0, pSize1, pSize2), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(new Shape(pSize0, pSize1), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(new Shape(pSize0), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(pShape, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Tensor *pTensor)" << '\n';
    #endif  // __DEBUG__

    m_aShape     = NULL;
    m_aLongArray = NULL;
    m_Device     = CPU;
    m_idOfDevice = -1;
    Alloc(pTensor);
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::~Tensor()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetShape()" << '\n';
    #endif  // __DEBUG__

    return m_aShape;
}

template<typename DTYPE> int Tensor<DTYPE>::GetRank() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetRank();
}

template<typename DTYPE> int Tensor<DTYPE>::GetDim(int pRanknum) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetDim(pRanknum);
}

template<typename DTYPE> LongArray<DTYPE> *Tensor<DTYPE>::GetLongArray() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetLongArray()" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray;
}

template<typename DTYPE> int Tensor<DTYPE>::GetCapacity() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCapacity()" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray->GetCapacity();
}

template<typename DTYPE> int Tensor<DTYPE>::GetElement(unsigned int index) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetElement(unsigned int index)" << '\n';
    #endif  // __DEBUG__

    return m_aLongArray->GetElement(index);
}

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::operator[](unsigned int index)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return (*m_aLongArray)[index];
}

template<typename DTYPE> Device Tensor<DTYPE>::GetDevice() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

template<typename DTYPE> IsUseTime Tensor<DTYPE>::GetIsUseTime() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetIsUseTime()" << '\n';
    #endif  // __DEBUG__

    return m_IsUseTime;
}

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetCPULongArray(unsigned int pTime) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCPULongArray(unsigned int pTime)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aLongArray->GetCPULongArray(pTime);
}

template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetTimeSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[0];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetBatchSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[1];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetChannelSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[2];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRowSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[3];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::GetColSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[4];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3 * pSize4;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(4, pSize0, pSize1, pSize2, pSize3);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(3, pSize0, pSize1, pSize2);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(2, pSize0, pSize1);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(1, pSize0);
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset()" << '\n';
    #endif  // __DEBUG__

    int capacity = GetCapacity();

    #ifdef __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    for (int i = 0; i < capacity; i++) {
        (*m_aLongArray)[i] = 0;
    }
}

template<typename DTYPE> void Tensor<DTYPE>::SetDeviceCPU() {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;
    m_aLongArray->SetDeviceCPU();
    m_aShape->SetDeviceCPU();
}

template<typename DTYPE> int Tensor<DTYPE>::Save(unsigned int idxOfParameter) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Save(FILE *fileForSave)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    m_aLongArray->Save(idxOfParameter);


    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Load(unsigned int idxOfParameter) {
    #ifdef __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Load(FILE *fileForSave)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    m_aLongArray->Load(idxOfParameter);

    return TRUE;
}

#ifdef __CUDNN__
template<typename DTYPE> void Tensor<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__
    checkCudaErrors(cudaSetDevice(idOfDevice));

    m_Device     = GPU;
    m_idOfDevice = idOfDevice;
    m_aLongArray->SetDeviceGPU(idOfDevice);
    m_aShape->SetDeviceGPU(idOfDevice);
}

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetGPUData(unsigned int pTime) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetGPUData(unsigned int pTime)" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetGPUData" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_aLongArray->GetGPUData(pTime);
}

template<typename DTYPE> cudnnTensorDescriptor_t& Tensor<DTYPE>::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDescriptor()" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        }
        this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : GetDescriptor" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    return m_aShape->GetDescriptor();
}

template<typename DTYPE> void Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle)" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");

        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : Reset" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        if (m_idOfDevice == -1) {
            std::cout << "you need to set device GPU first before : Reset" << '\n';
            exit(-1);
        } else this->SetDeviceGPU(m_idOfDevice);
    }

    # endif // __DEBUG__

    int pTime                     = this->GetTimeSize();
    cudnnTensorDescriptor_t pDesc = this->GetDescriptor();
    DTYPE *pDevLongArray          = NULL;
    float  zero                   = 0.f;

    for (int i = 0; i < pTime; i++) {
        pDevLongArray = this->GetGPUData(i);
        checkCUDNN(cudnnAddTensor(pCudnnHandle,
                                  &zero, pDesc, pDevLongArray,
                                  &zero, pDesc, pDevLongArray));
    }
}

#endif  // if __CUDNN__

////////////////////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Random_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Random_normal()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Random_normal(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), mean, stddev, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Random_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Random_normal()" << '\n';
    #endif  // __DEBUG__
    srand((unsigned)time(NULL));

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int   capacity = temp->GetCapacity();
    DTYPE v1 = 0.f, v2 = 0.f, mid_result = 0.f;

    // Random number generator on normal distribution
    for (int i = 0; i < capacity; i++) {
        do {
            v1         = 2 * ((float)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
            v2         = 2 * ((float)rand() / RAND_MAX) - 1; // -1.0 ~ 1.0 까지의 값
            mid_result = v1 * v1 + v2 * v2;
        } while (mid_result >= 1 || mid_result == 0);

        mid_result = sqrt((-2 * log(mid_result)) / mid_result);
        mid_result = v1 * mid_result;
        (*temp)[i] = (stddev * mid_result) + mean;
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Zeros(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(Shape *pShape, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return new Tensor<DTYPE>(pShape, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    return Tensor<DTYPE>::Constants(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), constant, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer) {
    #ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[1] + ch) * (*pShape)[2] + ro) * (*pShape)[3] + co;
}

inline unsigned int Index3D(Shape *pShape, int ch, int ro, int co) {
    return (ch * (*pShape)[1] + ro) * (*pShape)[2] + co;
}

inline unsigned int Index2D(Shape *pShape, int ro, int co) {
    return ro * (*pShape)[1] + co;
}

#endif  // TENSOR_H_
