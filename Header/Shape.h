#ifndef __SHAPE__
#define __SHAPE__    value

#include "Common.h"

class Shape {
private:
    int m_Rank;
    int *m_aDim;

public:
    Shape();
    Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    Shape(Shape *pShape);  // Copy Constructor
    virtual ~Shape();

    int  Alloc();
    int  Alloc(int pRank, ...);
    int  Alloc(Shape *pShape);
    void Delete();

    void SetRank(int pRank);
    int  GetRank();

    int& operator[](int pRanknum);

};
/////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& pOS, Shape *pShape);

#endif  // __SHAPE__
