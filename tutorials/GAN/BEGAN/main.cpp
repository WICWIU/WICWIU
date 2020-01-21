#include "net/my_BEGAN.hpp"
#include "CelebADataset.hpp"
#include <time.h>

#define BATCH                 64
#define EPOCH                 10000
#define LOOP_FOR_TRAIN        (202599 / BATCH)
#define LOOP_FOR_TEST         10//(10000 / BATCH)
#define LOOP_FOR_TRAIN_DISC   1
#define GPUID                 2

using namespace std;

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;
    char filename[]      = "GAN_params";
    string path = "../../../../../64_celebA";


    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *z     = new Tensorholder<float>(1, BATCH, 1, 1, 100, "z");
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 12288, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 1, "label");

    // create NoiseGenrator
    GaussianNoiseGenerator<float> *Gnoise = new GaussianNoiseGenerator<float>(1, BATCH, 1, 1, 100, 0, 1);

    // ======================= Select net ===================
    BEGAN<float> *net  = new my_BEGAN<float>(z, x, label);
    //net->Load(filename);

    // ======================= Prepare Data ===================
    // stenford cars data
    CelebADataset<float> *celebA = new CelebADataset<float>(path);
    // float mean[3] = {0.f, 0.f, 0.f};
    // float std[3] = {0.5, 0.5, 0.5};
    //
    // car->UseNormalization(TRUE, mean, std);

    DataLoader<float> *dl = new DataLoader<float>(celebA, BATCH, TRUE, 16, FALSE);


#ifdef __CUDNN__
    net->SetDeviceGPUOnGAN(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    float bestGenLoss  = 0.f;
    float bestDiscLoss = 0.f;
    int   epoch        = 0;

    // @ When load parameters
    // net->Load(filename);

    std::cout << "bestGenLoss : " << bestGenLoss << '\n';
    std::cout << "bestDiscLoss : " << bestDiscLoss << '\n';
    std::cout << "epoch : " << epoch << '\n';

    //Start make Noise
    Gnoise->StartProduce();

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Train =======================
        float genLoss  = 0.f;
        float discLoss = 0.f;

        net->SetModeTrain();
        startTime = clock();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            for(int k = 0; k < LOOP_FOR_TRAIN_DISC; k++){
                std::vector<Tensor<float> *> * temp = dl->GetDataFromGlobalBuffer();
                Tensor<float> *x_t = (*temp)[0];
                Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();
#ifdef __CUDNN__
                x_t->SetDeviceGPU(GPUID);
                z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
                net->FeedInputTensor(2, z_t, x_t);
                net->ResetParameterGradient();
                net->TrainDiscriminator();
            }

            Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();

#ifdef __CUDNN__
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
            net->FeedInputTensor(1, z_t);
            net->ResetParameterGradient();
            net->TrainGenerator();
            net->UpdateK();
            net->ComputeConvergenceMeasure();

            genLoss  = (*net->GetGeneratorLossFunction()->GetResult())[Index5D(net->GetGeneratorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
            discLoss  = (*net->GetDiscriminatorLossFunction()->GetResult())[Index5D(net->GetDiscriminatorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

            printf("Train complete percentage is %d / %d -> Generator Loss : %f, Discriminator Loss : %f\n",
                   j + 1,
                   LOOP_FOR_TRAIN,
                   genLoss,
                   discLoss);
            //  fflush(stdout);
             if(j % 50 == 0){
                 string filePath  = "generated/epoch" + std::to_string(i) + "_" + std::to_string(j) + ".jpg";
                 const char *cstr = filePath.c_str();
                 Tensor2Image<float>(net->GetGenerator()->GetResult(), cstr, 3, 64, 64);
            }
        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        net->Save(filename);
    }

    //Stop making Noise
    Gnoise->StopProduce();
    delete Gnoise;

    delete net;

    return 0;
}
