#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

template<typename DTYPE> class Data {
private:
    int m_timeSize;
    int m_capacityPerTime;  // max column size
    DTYPE **m_aHostData;
#if __CUDNN__
    DTYPE **m_aDevData;
#endif  // __CUDNN

    Device m_Device;

public:
    Data();
    Data(unsigned int pCapacity);
    Data(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    Data(Data *pData);  // Copy Constructor
    virtual ~Data();

    int    Alloc();
    int    Alloc(Data *pData);
    void   Delete();

    int    GetCapacity();
    int    GetTimeSize();
    int    GetCapacityPerTime();

    DTYPE* GetHostData(unsigned int pTime = 0);

#ifdef __CUDNN__
    DTYPE* GetDeviceData(unsigned int pTime = 0);
    void   MemcpyDeviceToHost();
    void   MemcpyHostToDevice();

#endif  // if __CUDNN__

    DTYPE& operator[](unsigned int index);
};


#endif  // __DATA__
