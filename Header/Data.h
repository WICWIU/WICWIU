#ifndef __DATA__
#define __DATA__      value

#include "Common.h"

#define SIZEOFCOLS    1024

template<typename DTYPE> class Data {
private:
    int m_Capacity;
    int m_Cols;  // max column size
    int m_Rows;
    DTYPE **m_aData;

public:
    Data();
    Data(unsigned int pCapacity);
    Data(Data *pData);  // Copy Constructor
    virtual ~Data();

    int    Alloc(unsigned int pCapacity);
    int    Alloc(Data *pData);
    void   Delete();

    int    GetCapacity();

    DTYPE& GetRawData();
    DTYPE& operator[](unsigned int index);
};


#endif  // __DATA__
