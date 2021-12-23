#ifndef __ENCODER__
#define __ENCODER__    value

#include "../Module.hpp"

template<typename DTYPE> class Encoder : public Module<DTYPE>{
private:
public:
    Encoder(std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pName);
    }

    virtual ~Encoder() {}

    int Alloc(std::string pName) {
        return TRUE;
    }
};

#endif  //__ENCODER__
