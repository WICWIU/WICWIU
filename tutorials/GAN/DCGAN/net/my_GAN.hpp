#include <iostream>
#include <string>

#include "my_Generator.hpp"
#include "my_Discriminator.hpp"
#include "../../../../WICWIU_src/GAN.hpp"

template<typename DTYPE> class my_GAN : public GAN<DTYPE> {
public:
    my_GAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_GAN() {
    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        this->SetInput(3, z, x, label);

        this->SetGenerator(new my_Generator<float>(z));
        this->SetSwitch(new Switch<float>(this->GetGenerator(), x));
        this->SetDiscriminator(new my_Discriminator<float>(this->GetSwitch()));
        this->AnalyzeGraph(this->GetDiscriminator());

        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        this->SetGANLossFunctions(new VanillaGANGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanillaGANGeneratorLoss"), new VanillaGANDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanillaGANDiscriminatorLoss"));

        // ======================= Select Optimizer ===================
        // this->SetGANOptimizers(new GradientDescentOptimizer<float>(this->GetGenerator()->GetParameter(), 0.000001, MINIMIZE), new GradientDescentOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.000001, MAXIMIZE));
        // this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MINIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MAXIMIZE));
        this->SetGANOptimizers(new AdamOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0002, 0.5, 0.999, 1e-08, MINIMIZE), new AdamOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.00007, 0.5, 0.999, 1e-08, MINIMIZE));
    }
};
