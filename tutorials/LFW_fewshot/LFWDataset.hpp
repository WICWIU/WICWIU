#ifndef __LFWDataset__
#define __LFWDataset__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "../../WICWIU_src/DataLoader.hpp"
#include "ImageProcess.hpp"

//#define NUMBER_OF_CLASS               143
//#define NUMBER_OF_CLASS               5749
#define NUMBER_OF_CHANNEL             3
#define LEGNTH_OF_WIDTH_AND_HEIGHT    220
#define IMAGE_SIZE             48400
#define INPUT_DIM             145200

#define __TURBOJPEG__

#ifdef __TURBOJPEG__
# include <turbojpeg.h>
#endif  // ifdef __TURBOJPEG__

template<typename DTYPE>
class LFWDataset : public Dataset<DTYPE>{
private:
    int m_numOfImg;
    std::string m_rootPath;
    std::string m_dataPath;
    vision::Compose *m_transform;
    sem_t sem;

    // set of name of Class
    int m_useClasNum;
    std::vector<std::string> m_className;
    std::vector<std::string> m_aImagePath;
    std::vector<int> m_vSamplePerClass;
//    std::vector<int> m_aLabel;
    int trigger;
    int imgNum[20];
//    int count;
//    int check;

    void           CheckClassList();
    void           CreateImageListOfEachClass();
    void           CountSamplePerClass(int maxClass = 0);

#ifdef __TURBOJPEG__
    void           AllocImageBuffer(int idx, ImageWrapper& imgWrp);
    void           DeleteImageBuffer(ImageWrapper& imgWrp);
    Tensor<DTYPE>* Image2Tensor(ImageWrapper& imgWrp, int doValueScaling);

#endif  // ifdef __TURBOJPEG__

public:
    LFWDataset(std::string rootPath, std::string dataPath, int useClassNum, vision::Compose *transform) {
        m_rootPath   = rootPath;
        m_dataPath   = dataPath;
        m_useClasNum = useClassNum;
        m_numOfImg = 0;
        trigger      = 0;

//        assert((useClassNum > 0) && (useClassNum <= NUMBER_OF_CLASS));
        m_transform  = transform;
        assert(m_transform != NULL);

        Alloc();
        CheckClassList();
        CreateImageListOfEachClass();

        LogMessageF("lfw_funneled_label.txt", TRUE, "%d samples\n", this->GetLength());
#ifdef  __DEBUG__        
        for(int i = 0; i < this->GetLength(); i++)
            LogMessageF("lfw_funneled_label.txt", FALSE, "%d\t%d\n", i, this->GetLabel(i));
        // MyPause(__FUNCTION__);
#endif//  __DEBUG__            

        CountSamplePerClass();
    }

    virtual ~LFWDataset() {
        Delete();
    }

    virtual void                          Alloc();

    virtual void                          Delete();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

//    virtual int                           GetLength();

    virtual void                          CopyData(int idx, DTYPE *pDest);             // copy i-th iamge into pDest. (designed for k-NN)

    std::string& GetImagePath(int idx)   { return m_aImagePath[idx]; }

    int GetSampleCount(int classId)     { return (classId >= 0 && classId < m_vSamplePerClass.size()) ? m_vSamplePerClass[classId] : 0;}
    int GetNoMinorClass(int minCount = 2);

    void Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling);
};

//

template<typename DTYPE> void LFWDataset<DTYPE>::Alloc()  { sem_init(&sem, 0, 1);}

template<typename DTYPE> void LFWDataset<DTYPE>::Delete() {sem_destroy(&sem);}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *LFWDataset<DTYPE>::GetData(int idx)
{
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    Tensor<DTYPE> *image = NULL;
    Tensor<DTYPE> *label = NULL;
    ImageWrapper   imgWrp;

    // image load
#ifdef __TURBOJPEG__
    this->AllocImageBuffer(idx, imgWrp);
    // std::cout << imgWrp.imgShape << '\n';
#endif  // ifdef __TURBOJPEG__

    // if(m_option == "train") else if(m_option == "test") else exit(-1);
    m_transform->DoTransform(imgWrp);
    // std::cout << imgWrp.imgShape << '\n';

    image = this->Image2Tensor(imgWrp, TRUE);
    // std::cout << image->GetShape() << '\n';
    // DisplayFeature(100, &(*image)[0]);
    // this->Tensor2Image("test.jpeg", image, TRUE);
    result->push_back(image);

    // push label as (one-hot vector)
    label                   = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_useClasNum);
    (*label)[this->GetLabel(idx)] = (DTYPE)1;    
    result->push_back(label);

    return result;
}

template<typename DTYPE> void LFWDataset<DTYPE>::CopyData(int idx, DTYPE *pDest)
{
//    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);
#ifdef  __DEBUG__
    static int called = 0;
    printf("%s, called = %d\n", __FUNCTION__, called++);
    fflush(stdout);
    if(called == 188)
        printf("hi!");
#endif  // __DEBUG__

    Tensor<DTYPE> *image = NULL;
//    Tensor<DTYPE> *label = NULL;
    ImageWrapper   imgWrp;

    // image load
#ifdef __TURBOJPEG__
    this->AllocImageBuffer(idx, imgWrp);
    // std::cout << imgWrp.imgShape << '\n';
#endif  // ifdef __TURBOJPEG__

    // if(m_option == "train") else if(m_option == "test") else exit(-1);
    m_transform->DoTransform(imgWrp);
    // std::cout << imgWrp.imgShape << '\n';

    image = this->Image2Tensor(imgWrp, TRUE);
//  std::cout << image->GetShape() << '\n';
//    DisplayFeature(100, &(*image)[0]);
//    this->Tensor2Image("test.jpeg", image, TRUE);

    memcpy(pDest, &(*image)[0], INPUT_DIM * sizeof(DTYPE));

    delete image;
}   

template<typename DTYPE> int LFWDataset<DTYPE>::GetNoMinorClass(int minCount)
{
    int n = 0;
    for(int i = 0; i < m_useClasNum; i++){
        if(this->GetSampleCount(i) < minCount)
            n++;
    }

    return n;
}

template<typename DTYPE> void LFWDataset<DTYPE>::CheckClassList() {
    // mnt/ssd/Data/ImageNet/synset_words.txt
    std::string filePath = m_rootPath + "/" + m_dataPath + ".txt";
    const char *cstr     = filePath.c_str();

    FILE *pFile = NULL;

    pFile = fopen(cstr, "r");

    if (pFile == NULL) {
        printf("Failed to open file %s in %s\n", cstr, __FUNCTION__);
        exit(-1);
    } else {
        char realValue[100];

        for (int i = 0; i < m_useClasNum; i++) {
            if (fscanf(pFile, "%s", realValue)) {
                m_className.push_back((std::string)realValue);
                while (fgetc(pFile) != '\n') ;
            } else {
                printf("there is something error\n");
                exit(-1);
            }
        }
    }

    fclose(pFile);
}

template<typename DTYPE> void LFWDataset<DTYPE>::CreateImageListOfEachClass() {
    // list file : 1st line - number of image, the others - image file name

    std::vector<int> vTmpLabel;
    vTmpLabel.reserve(2048);

    for (int classNum = 0; classNum < m_useClasNum; classNum++) {
        std::string filePath = m_rootPath + '/' + m_dataPath + '/' + m_className[classNum] + "/list.txt";  // check with printf
        const char *cstr     = filePath.c_str();

        FILE *pFile = NULL;
        pFile = fopen(cstr, "r");

        char realValue[100];
        int  numOfImageOfClass = 0;

        if (pFile == NULL) {
            printf("Failed to open file %s in %s\n", cstr, __FUNCTION__);
            exit(-1);
        } else {
            if (fscanf(pFile, "%s", realValue)) {  // first realValue is already readed above
                numOfImageOfClass = atoi(realValue);

                for (int imageNum = 0; imageNum < numOfImageOfClass; imageNum++) {
                    if (fscanf(pFile, "%s", realValue)) {
                        m_aImagePath.push_back((std::string)(m_className[classNum] + '/' + realValue));
                        vTmpLabel.push_back(classNum);
                        // std::cout << m_aImagePath.back() << " : " << m_aLabel.back()  << " : " << m_aLabel.size() << '\n';
                    } else {
                        printf("there is something error\n");
                        exit(-1);
                    }
                }
            } else {
                printf("there is something error\n");
                exit(-1);
            }
        }
        fclose(pFile);
    }

    m_numOfImg = m_aImagePath.size();

    this->SetLabel(&vTmpLabel[0], m_numOfImg);

    assert(m_numOfImg > 0);
    // std::cout << "m_numOfImg : " << m_numOfImg << '\n';
}

template<typename DTYPE> void LFWDataset<DTYPE>::CountSamplePerClass(int maxClass)
{
    int noSample = this->GetLength();

    if(maxClass < 0)
        maxClass = 0;

    m_vSamplePerClass.resize(maxClass);

    // not necessary
    // if(maxClass > 0)
    //     memset(&m_vSamplePerClass[0], 0, maxClass * sizeof(m_vSamplePerClass[0]));

    for(int i = 0; i < noSample; i++){
        int classId = this->GetLabel(i);
        if(classId >= m_vSamplePerClass.size()){
            int oldSize = m_vSamplePerClass.size();
            m_vSamplePerClass.resize(classId + 1);
// not necessary
//            memset(&m_vSamplePerClass[oldSize], 0, (classId - oldSize) * sizeof(m_vSamplePerClass[0]));
        }

        m_vSamplePerClass[classId]++;        
    }
}

#ifdef __TURBOJPEG__

template<typename DTYPE> void LFWDataset<DTYPE>::AllocImageBuffer(int idx, ImageWrapper& imgWrp) {
    int   width, height;
    FILE *jpegFile         = NULL;
    unsigned char *jpegBuf = NULL;
    int pixelFormat        = TJPF_RGB;
    tjhandle tjInstance    = NULL;
    long     size;
    int inSubsamp, inColorspace;
    unsigned long jpegSize;

    // create file address
    std::string filePath = m_rootPath + '/' + m_dataPath + '/' + m_aImagePath[idx];  // check with printf
    const char *cstr     = filePath.c_str();

    // Load image (no throw and catch)
    /* Read the JPEG file into memory. */
    jpegFile = fopen(cstr, "rb");

    fseek(jpegFile, 0, SEEK_END);
    size = ftell(jpegFile);
    fseek(jpegFile, 0, SEEK_SET);

    jpegSize = (unsigned long)size;
    jpegBuf  = (unsigned char *)tjAlloc(jpegSize);

    if (fread(jpegBuf, jpegSize, 1, jpegFile) < 1) exit(-1);

    fclose(jpegFile);
    jpegFile = NULL;

    tjInstance = tjInitDecompress();
    tjDecompressHeader3(tjInstance, jpegBuf, jpegSize, &width, &height, &inSubsamp, &inColorspace);
    imgWrp.imgBuf = (unsigned char *)tjAlloc(width * height * tjPixelSize[pixelFormat]);
    tjDecompress2(tjInstance, jpegBuf, jpegSize, imgWrp.imgBuf, width, 0, height, pixelFormat, 0);

    imgWrp.imgShape = new Shape(tjPixelSize[pixelFormat], height, width);

    tjFree(jpegBuf);
    jpegBuf = NULL;
    tjDestroy(tjInstance);
    tjInstance = NULL;
}

template<typename DTYPE> Tensor<DTYPE> *LFWDataset<DTYPE>::Image2Tensor(ImageWrapper& imgWrp, int doValueScaling) {
    unsigned char *imgBuf = imgWrp.imgBuf;
    Shape *imgShape       = imgWrp.imgShape;

    int width   = imgShape->GetDim(2);
    int height  = imgShape->GetDim(1);
    int channel = imgShape->GetDim(0);

    Tensor<DTYPE> *result = Tensor<DTYPE>::Zeros(1, 1, channel, height, width);

    if (doValueScaling) {
        for (int ro = 0; ro < height; ro++) {
            for (int co = 0; co < width; co++) {
                for (int ch = 0; ch < channel; ch++) {
                    (*result)[Index5D(result->GetShape(), 0, 0, ch, ro, co)] = imgBuf[ro * width * channel + co * channel + ch] / 255.0;
                }
            }
        }
    } else {
        for (int ro = 0; ro < height; ro++) {
            for (int co = 0; co < width; co++) {
                for (int ch = 0; ch < channel; ch++) {
                    (*result)[Index5D(result->GetShape(), 0, 0, ch, ro, co)] = imgBuf[ro * width * channel + co * channel + ch];
                }
            }
        }
    }

    return result;
}

template<typename DTYPE> void LFWDataset<DTYPE>::Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling) {
    int width   = imgTensor->GetShape()->GetDim(4);
    int height  = imgTensor->GetShape()->GetDim(3);
    int channel = imgTensor->GetShape()->GetDim(2);

    unsigned char *imgBuf   = new unsigned char[channel * height * width];
    int pixelFormat         = TJPF_RGB;
    unsigned char *jpegBuf  = NULL;  /* Dynamically allocate the JPEG buffer */
    unsigned long  jpegSize = 0;
    FILE *jpegFile          = NULL;
    tjhandle tjInstance     = NULL;

    if (imgTensor) {
        if (doValuerScaling) {
            for (int ro = 0; ro < height; ro++) {
                for (int co = 0; co < width; co++) {
                    for (int ch = 0; ch < channel; ch++) {
                        imgBuf[ro * width * channel + co * channel + ch] = (*imgTensor)[Index5D(imgTensor->GetShape(), 0, 0, ch, ro, co)] * 255.0;
                    }
                }
            }
        } else {
            for (int ro = 0; ro < height; ro++) {
                for (int co = 0; co < width; co++) {
                    for (int ch = 0; ch < channel; ch++) {
                        imgBuf[ro * width * channel + co * channel + ch] = (*imgTensor)[Index5D(imgTensor->GetShape(), 0, 0, ch, ro, co)];
                    }
                }
            }
        }

        tjInstance = tjInitCompress();
        tjCompress2(tjInstance, imgBuf, width, 0, height, pixelFormat,
                    &jpegBuf, &jpegSize,  /*outSubsamp =*/ TJSAMP_444,  /*outQual =*/ 100,  /*flags =*/ 0);
        tjDestroy(tjInstance);
        tjInstance = NULL;
        delete imgBuf;

        if (!(jpegFile = fopen(filename.c_str(), "wb"))) {
            printf("Failed to open file %s in %s\n", filename.c_str(), __FUNCTION__);
            exit(-1);
        } else {
            fwrite(jpegBuf, jpegSize, 1, jpegFile);
        }

        if (jpegFile) {
            fclose(jpegFile);
            jpegFile = NULL;
        }

        if (jpegBuf) {
            tjFree(jpegBuf);
            jpegBuf = NULL;
        }
    } else {
        printf("Invalid Tensor pointer");
        exit(-1);
    }
}

#endif  // ifdef __TURBOJPEG__


#endif  // __LFWDataset__