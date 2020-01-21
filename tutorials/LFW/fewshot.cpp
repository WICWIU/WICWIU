#include <stdio.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <fstream>

#include <unistd.h>

#include <Utils.hpp>

#include "net/my_Facenet.hpp"
#include "LFWDataset.hpp"
#include "../../WICWIU_src/KNearestNeighbor.hpp"
#include "../../WICWIU_src/FewShotClassifier.hpp"

#define CNN_OUTPUT                  5749
#define NUMBER_OF_CLASS               10
//#define NUMBER_OF_CLASS               11
#define BATCH                         45
#define NEW_CLASS_INDEX                     149     // start index of Jeremy_Greenstock is 150
#define GPUID                         0
//#define GPUID                         1
#define LOG_LENGTH                    1
#define SAMPLE_PER_CLASS              5
#define KNN_K                         3
// #define IMAGE_SIZE                   48400      // 220*220, defined in LFWDataset.hpp
// #define INPUT_DIM                   145200      // 3 * 220 * 200, defined in LFWDataset.hpp
#define FEATURE_DIM                 128

// #define ENABLE_TRAINING
#define ENABLE_TEST


void GetReferenceSamples(LFWDataset<float> *dataset, int dim, int noClass, int samplePerClass, std::vector<float*> &vSamples, std::vector<int> &vLabels, std::vector<int> *pvRefIndex = NULL, int startIdx = 0);        // designed for k-NN
int CopyFile(const char *srcFile, const char *destFile);

int main(int argc, char const *argv[]) {
    time_t startTime;
    struct tm *curr_tm;
    double     nProcessExcuteTime;

    char filename[] = "LFW_params";
    int num = 0;

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 145200, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, NUMBER_OF_CLASS, "label"); // NUMBER_OF_CLASS is not used

    // ======================= Creating net ===================
    printf("Creating neural net...");   fflush(stdout);
    NeuralNetwork<float> *net = new my_FaceNet<float>(x, label, NUMBER_OF_CLASS);   // NUMBER_OF_CLASS is not used
    printf("Done.\n");   fflush(stdout);
    // net->PrintGraphInformation();


    // Load parameters from model file
    if(access(filename, 00) == 0){
        net->Load(filename);
        printf("Parameters loaded from %s\n", filename);
        fflush(stdout);
    } else  {
        printf("Failed to load parameters from %s\n", filename);
        fflush(stdout);
        exit(-1);
    }

#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    // std::cout << "/* Prepare Data */" << '\n';
    // ======================= Prepare Data ===================
    vision::Compose *transform = new vision::Compose({new vision::Resize(224), new vision::CenterCrop(220)});
    LFWDataset<float> *train_dataset = new LFWDataset<float>("./data", "ETRI_LFW_train", NUMBER_OF_CLASS, transform);
    LFWDataset<float> *test_dataset = new LFWDataset<float>("./data", "ETRI_LFW_test", NUMBER_OF_CLASS + 1, transform);

    // get referece samples and labels for few-shot classifier
    std::vector<float*> vRefSamples;
    std::vector<int> vRefLabels;

    std::vector<float*> test_vRefSamples;
    std::vector<int> test_vRefLabels;



    //for FewShotClassifier demo
    std::vector<std::string> before_classname = {"Bill_Gates", "George_Robertson", "Gray_Davis", "Halle_Berry", "Hans_Blix", "Hugo_Chavez", "Igor_Ivanov", "Jacques_Chirac", "Jean_Charest", "Jennifer_Aniston"};
    std::vector<std::string> after_classname = {"Bill_Gates", "George_Robertson", "Gray_Davis", "Halle_Berry", "Hans_Blix", "Hugo_Chavez", "Igor_Ivanov", "Jacques_Chirac", "Jean_Charest", "Jennifer_Aniston", "Jeremy_Greenstock"};
    std::ifstream fin;
    std::ofstream fout;
    net->SetModeInference();

    LFWDataset<float> &dataset = *test_dataset;

    printf("Creating Few-Shot Classifier with %d classes\n", NUMBER_OF_CLASS);
    GetReferenceSamples(train_dataset, INPUT_DIM, NUMBER_OF_CLASS, SAMPLE_PER_CLASS, vRefSamples, vRefLabels);
    FewShotClassifier few_shot(INPUT_DIM, FEATURE_DIM, before_classname, net, vRefSamples.size(), &vRefLabels[0], &vRefSamples[0], BATCH);

    std::vector<float> testSample(INPUT_DIM);
    std::string result;

    printf("Recognizing image of Jeremy_Greenstock, who is NOT in the target classes.\n");
    int testIdx = 6;
    do {
        printf("Select a sample %d ~ 15 : ", SAMPLE_PER_CLASS);
        scanf("%d", &testIdx);
    } while(testIdx < 1 || testIdx > 15);

    char inputFile[256];
    sprintf(inputFile, "./data/ETRI_LFW_test/Jeremy_Greenstock/Jeremy_Greenstock_%04d.jpg", testIdx);
    CopyFile(inputFile, "image_input.jpg");

    dataset.CopyData(NEW_CLASS_INDEX + testIdx, &testSample[0]);
    printf("Recognizing %d-th sample of Jeremy_Greenstock.........\n", testIdx);
    result = few_shot.Recognize(&testSample[0], KNN_K);

    printf("Original label: Jeremy_Greenstock\n");
    printf("Recognition result: %s\n", result.c_str());

    std::string img_name = "./data/ETRI_LFW_test/" + result + "/" + result + "_0001.jpg";
    CopyFile(img_name.c_str(), "image_result (before).jpg");
    printf("Reference image of recognition result is saved in 'image_result (before).jpg'\n");

    // printf("Press Enter to add a new class Jeremy_Greenstock...");
    // getchar();

    for(int i = 0; i < SAMPLE_PER_CLASS; i++){
        printf("Adding %d-th sample of Jeremy_Greenstock...\n", i + 1);     fflush(stdout);
        dataset.CopyData(NEW_CLASS_INDEX + i, &testSample[0]);
        few_shot.AddReference("Jeremy_Greenstock", &testSample[0]);
    }
    printf("Done.\n");

    // printf("Make KNN reference with Testing dataset with New Class\n");
    // GetReferenceSamples(test_dataset, INPUT_DIM, NUMBER_OF_CLASS + 1, SAMPLE_PER_CLASS, test_vRefSamples, test_vRefLabels);
    // FewShotClassifier test_few_shot(INPUT_DIM, FEATURE_DIM, after_classname, net, test_vRefSamples.size(), &test_vRefLabels[0], &test_vRefSamples[0], BATCH);

    std::vector<float*> test_vTestSample;

    AllocFeatureVector(INPUT_DIM, 1, test_vTestSample);

    dataset.CopyData(NEW_CLASS_INDEX + testIdx, test_vTestSample[0]);
//    result = test_few_shot.Recognize(test_vTestSample[0], KNN_K);
    result = few_shot.Recognize(test_vTestSample[0], KNN_K);

    DeleteFeatureVector(test_vTestSample);

    std::cout << "Original label: Jeremy_Greenstock" << '\n';
    std::cout << "New Testing set Few_shot result: " << result <<'\n';
    img_name = "./data/ETRI_LFW_test/" + result + "/" + result + "_0001.jpg";

    CopyFile(img_name.c_str(), "image_result (after).jpg");
    printf("Reference image of recognition result is saved in 'image_result (before).jpg'\n");

    delete net;
    delete train_dataset;
    delete test_dataset;

    return 0;
}

void GetReferenceSamples(LFWDataset<float> *dataset, int dim, int noClass, int samplePerClass, std::vector<float*> &vSamples, std::vector<int> &vLabels, std::vector<int> *pvRefIndex, int startIdx)        // designed for k-NN
{
	DeleteFeatureVector(vSamples);
	vLabels.resize(0);
	if(pvRefIndex != NULL)
		pvRefIndex->resize(0);

	if(startIdx = -1)
		startIdx = rand() % dataset->GetLength();

    std::vector<int> vCount;
    vCount.resize(noClass);
    for(int i = 0; i < noClass; i++)
        vCount[i] = 0;

    int noComplete = 0;
    for(int i = 0; i < dataset->GetLength() && noComplete < noClass; i++){
		int idx = (startIdx + i) % dataset->GetLength();
        int label = dataset->GetLabel(idx);
        if(label >= noClass){
            printf("Error! label = %d, noClass = %d\n", label, noClass);
            printf("Press Enter to continue (%s)...", __FUNCTION__);    fflush(stdout);
            getchar();
        }
        if(vCount[label] < samplePerClass){
            vLabels.push_back(label);
			if(pvRefIndex)
				pvRefIndex->push_back(idx);

            float *pSample = new float[dim];
            if(pSample == NULL){
                printf("Failed to allocate memory, dim = %d in %s (%s %d)\n", dim, __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }

            dataset->CopyData(idx, pSample);
            vSamples.push_back(pSample);

            vCount[label]++;
            if(vCount[label] == samplePerClass)
                noComplete++;
        }
    }

#ifdef  __DEBUG__
    for(int i = 0; i < vLabels.size(); i++){
        printf("vLabels[%d] = %d\n", i, vLabels[i]);
    }

    // printf("Press Enter to continue...");
    // getchar();
#endif  // __DEBUG__
}

int CopyFile(const char *srcFile, const char *destFile)
{
    FILE *src = fopen(srcFile, "rb");
    if(src == NULL){
        printf("Failed to open %s in %s.\n", srcFile, __FUNCTION__);
        return FALSE;
    }

    FILE *dest = fopen(destFile, "wb");
    if(dest == NULL){
        printf("Failed to open %s in %s.\n", destFile, __FUNCTION__);
        fclose(src);
        return FALSE;
    }

    int blockSize = 1024 * 1024;
    unsigned char *buffer = new unsigned char [blockSize];
    if(buffer == NULL){
        printf("Failed to allocate memory in %s\n", __FUNCTION__);
        fclose(src);
        fclose(dest);
        return FALSE;
    }

    int ret = 0;
    while(1){
        ret = fread(buffer, 1, blockSize, src);
        if(ret == 0)
            break;
        fwrite(buffer, 1, ret, dest);
    } while (ret == blockSize);

    delete[] buffer;

    fclose(dest);
    fclose(src);

    return TRUE;
}
