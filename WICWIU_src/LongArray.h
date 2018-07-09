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

#ifdef __CUDNN__
    DTYPE **m_aaDevLongArray;
#endif  // __CUDNN

private:
    int  Alloc(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    int  Alloc(LongArray *pLongArray);
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
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
    DTYPE* GetCPULongArray(unsigned int pTime = 0);

    int    SetDeviceCPU();
#ifdef __CUDNN__
    int    SetDeviceGPU();

    DTYPE* GetGPUData(unsigned int pTime = 0);

#endif  // if __CUDNN__
};


#endif  // __DATA__
