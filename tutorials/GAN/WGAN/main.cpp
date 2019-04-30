#include "net/my_GAN.hpp"
#include "MNIST_Reader.hpp"
#include <time.h>

#define BATCH                 64
#define EPOCH                 200
#define LOOP_FOR_TRAIN        (60000 / BATCH)
#define LOOP_FOR_TEST         (10000 / BATCH)
#define LOOP_FOR_TRAIN_DISC   5
#define GPUID                 3

using namespace std;

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;
    char filename[]      = "GAN_params";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *z     = new Tensorholder<float>(1, BATCH, 1, 1, 100, "z");
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 1, "label");

    // create NoiseGenrator
    GaussianNoiseGenerator<float> *Gnoise = new GaussianNoiseGenerator<float>(1, BATCH, 1, 1, 100, 0, 1);

    // ======================= Select net ===================
    // GAN<float> *net  = new my_BEGAN<float>(z, x, label);
    GAN<float> *net  = new my_GAN<float>(z, x, label);
    //net->Load(filename);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

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
            dataset->CreateTrainDataPair(BATCH);

            Tensor<float> *x_t = dataset->GetTrainFeedImage();
            Tensor<float> *l_t = dataset->GetTrainFeedLabel();
            delete l_t;
            Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();

#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
            net->FeedInputTensor(2, z_t, x_t);
            net->ResetParameterGradient();
            net->TrainDiscriminator();
            net->Clip(-0.01, 0.01);

            z_t = Gnoise->GetNoiseFromBuffer();

#ifdef __CUDNN__
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__
            net->FeedInputTensor(1, z_t);
            net->ResetParameterGradient();
            net->TrainGenerator();

            genLoss  = (*net->GetGeneratorLossFunction()->GetResult())[Index5D(net->GetGeneratorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
            discLoss  = (*net->GetDiscriminatorLossFunction()->GetResult())[Index5D(net->GetDiscriminatorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

            printf("\rTrain complete percentage is %d / %d -> Generator Loss : %f, Discriminator Loss : %f",
                   j + 1,
                   LOOP_FOR_TRAIN,
                   genLoss,
                   discLoss);
             fflush(stdout);
             if(j % 50 == 0){
                 string filePath  = "generated/epoch" + std::to_string(i) + "_" + std::to_string(j) + ".jpg";
                 const char *cstr = filePath.c_str();
                 Tensor2Image<float>(net->GetGenerator()->GetResult()->GetResult(), cstr, 3, 20, 28, 28);
             }
        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test(Save Generated Image)======================
        float testGenLoss  = 0.f;
        float testDiscLoss = 0.f;

        net->SetModeInference();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();

#ifdef __CUDNN__
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            net->FeedInputTensor(1, z_t);
            net->Test();

            testGenLoss  = (*net->GetGeneratorLossFunction()->GetResult())[Index5D(net->GetGeneratorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
            testDiscLoss  = (*net->GetDiscriminatorLossFunction()->GetResult())[Index5D(net->GetDiscriminatorLossFunction()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];

            string filePath  = "evaluated/epoch" + std::to_string(i) + "_" + std::to_string(j) + ".jpg";
            const char *cstr = filePath.c_str();
            Tensor2Image<float>(net->GetGenerator()->GetResult()->GetResult(), cstr, 3, 20, 28, 28);

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1,
                   LOOP_FOR_TEST,
                   testGenLoss,
                   testDiscLoss);
            fflush(stdout);
        }
        std::cout << "\n\n";

        net->Save(filename);
        // Global Optimal C(G) = -log4
        // if ( abs(- 1.0 * log(4) - bestGenLoss) > abs(- 1.0 * log(4) - testGenLoss) ) {
        //     net->Save(filename);
        // }
    }

    //Stop making Noise
    Gnoise->StopProduce();
    delete Gnoise;

    delete dataset;
    delete net;

    return 0;
}
