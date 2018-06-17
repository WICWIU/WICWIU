#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

template<typename DTYPE> class Data {
private:
    DTYPE **m_aaHostData;

    int m_CapacityOfData;
    int m_TimeSize;
    int m_CapacityPerTime;

    Device m_Device;

#ifdef __CUDNN__
    DTYPE **m_aaDevData;
#endif  // __CUDNN

private:
    int  Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    int  Alloc(Data *pData);
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
    int  MemcpyCPU2GPU();
    int  MemcpyGPU2CPU();
#endif  // __CUDNN

public:
    Data(unsigned int pCapacity);
    Data(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    Data(Data *pData);  // Copy Constructor
    virtual ~Data();

    int    GetCapacity();
    int    GetTimeSize();
    int    GetCapacityPerTime();
    DTYPE  GetElement(unsigned int index);
    DTYPE& operator[](unsigned int index);
    Device GetDevice();
    DTYPE* GetCPUData(unsigned int pTime = 0);

    int    SetDeviceCPU();
#ifdef __CUDNN__
    int    SetDeviceGPU();

    DTYPE* GetGPUData(unsigned int pTime = 0);

#endif  // if __CUDNN__
};


#endif  // __DATA__
