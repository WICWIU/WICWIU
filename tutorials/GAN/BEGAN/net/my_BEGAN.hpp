#include <iostream>
#include <string>

#include "../../../../WICWIU_src/BEGAN.hpp"
#include "my_Generator.hpp"
#include "my_Discriminator.hpp"

template<typename DTYPE> class my_BEGAN : public BEGAN<DTYPE> {
private:
public:
    my_BEGAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_BEGAN() {

    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        this->SetInput(3, z, x, label);

        this->SetBEGANParameter(0.f, 0.001, 0.75);

        this->SetGenerator(new my_Generator<float>(z));
        this->SetSwitch(new Switch<float>(this->GetGenerator(), x));
        this->SetDiscriminator(new my_Discriminator<float>(this->GetSwitch()));
        this->AnalyzeGraph(this->GetDiscriminator());

        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        this->SetGANLossFunctions(new BEGANGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "BEGANGeneratorLoss"), new BEGANDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "BEGANDiscriminatorLoss"));
        // this->SetGANLossFunctions(new WGANGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "BEGANGeneratorLoss"), new WGANDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "BEGANDiscriminatorLoss"));

        // ======================= Select Optimizer ===================
        // this->SetGANOptimizers(new GradientDescentOptimizer<float>(this->GetGenerator()->GetParameter(), 0.01, MINIMIZE), new GradientDescentOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.001, MINIMIZE));
        // this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.001, 0.9, 1e-08, FALSE, MINIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MINIMIZE));
        this->SetGANOptimizers(new AdamOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0002, 0.5, 0.999, 1e-08, MINIMIZE), new AdamOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.001, 0.5, 0.999, 1e-08, MINIMIZE));
    }
};
