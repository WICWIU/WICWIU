#include "../../WICWIU_src/DataLoader.hpp"

template<typename DTYPE> class MNIST : public Dataset<DTYPE>{
private:
    /* data */

public:
    MNIST();
    virtual ~MNIST();
    virtual void                          Alloc();
    virtual void                          Dealloc();
    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);
    virtual int                           GetLength();
};
