#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

template<typename DTYPE> class Data {
private:
    int m_timeSize;
    int m_capacityPerTime;  // max column size
    DTYPE **m_aData;

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

    DTYPE* GetLowData(unsigned int pTime = 0);
    DTYPE& operator[](unsigned int index);
};


#endif  // __DATA__
