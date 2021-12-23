#ifndef __DECODER__
#define __DECODER__    value

#include "../Module.hpp"

template<typename DTYPE> class Decoder : public Module<DTYPE>{
private:
public:
    Decoder(std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pName);
    }

    virtual ~Decoder() {}

    int Alloc(std::string pName) {
        return TRUE;
    }

};
#endif  // __DECODER__
