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

#define __TURBOJPEG__

// #ifdef __TURBOJPEG__
// # include <turbojpeg.h>
// #endif  // ifdef __TURBOJPEG__

#include "../../WICWIU_src/DataLoader.hpp"
#include "ImageProcess.hpp"

#define NUMBER_OF_CLASS               1000
#define NUMBER_OF_CHANNEL             3
#define LEGNTH_OF_WIDTH_AND_HEIGHT    224
#define CAPACITY_OF_PLANE             50176
#define CAPACITY_OF_IMAGE             150528

#define  TRAIN_FOLDER                 "ILSVRC2012_img_train256";
#define  TEST_FOLDER                  "ILSVRC2012_img_val256";

#define __TURBOJPEG__

#ifdef __TURBOJPEG__
# include <turbojpeg.h>
#endif  // ifdef __TURBOJPEG__


template<typename DTYPE>
class ImageNetDataset : public Dataset<DTYPE>{
private:
    int m_numOfImg;
    std::string m_rootPath;
    std::string m_dataPath;
    vision::Compose *m_transform;

    // set of name of Class
    int m_useClasNum;
    std::vector<std::string> m_className;
    int m_numOfImage;
    std::vector<std::string> m_aImage;
    std::vector<int> m_aLable;

    void           CheckClassList();
    void           CreateImageListOfEachClass();

#ifdef __TURBOJPEG__
    void           AllocImageBuffer(int idx, ImageWrapper& imgWrp);
    void           DeleteImageBuffer(ImageWrapper& imgWrp);
    Tensor<DTYPE>* Image2Tensor(ImageWrapper& imgWrp, int doValueScaling);

#endif  // ifdef __TURBOJPEG__

public:
    ImageNetDataset(std::string rootPath, std::string dataPath, int useClassNum, vision::Compose *transform) {
        m_rootPath   = rootPath;
        m_dataPath   = dataPath;
        m_useClasNum = useClassNum;
        assert((useClassNum > 0) && (useClassNum <= NUMBER_OF_CLASS));
        m_transform  = transform;
        assert(m_transform != NULL);

        Alloc();
        CheckClassList();
        CreateImageListOfEachClass();
    }

    virtual ~ImageNetDataset() {
        Delete();
    }

    virtual void                          Alloc();

    virtual void                          Delete();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();
    void Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling);

};

//

template<typename DTYPE> void ImageNetDataset<DTYPE>::Alloc()  {}

template<typename DTYPE> void ImageNetDataset<DTYPE>::Delete() {}

//

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *ImageNetDataset<DTYPE>::GetData(int idx) {
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
    // this->Tensor2Image("test.jpeg", image, TRUE);
    result->push_back(image);

    label                   = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_useClasNum);
    (*label)[m_aLable[idx]] = (DTYPE)1;
    result->push_back(label);

    return result;
}

template<typename DTYPE> int ImageNetDataset<DTYPE>::GetLength() {
    return m_numOfImage;
}

template<typename DTYPE> void ImageNetDataset<DTYPE>::CheckClassList() {
    // mnt/ssd/Data/ImageNet/synset_words.txt
    std::string filePath = m_rootPath + "/synset_words.txt";
    const char *cstr     = filePath.c_str();

    // std::cout << filePath << '\n';

    FILE *pFile = NULL;

    pFile = fopen(cstr, "r");

    if (pFile == NULL) {
        printf("file open fail\n");
        exit(-1);
    } else {
        char realValue[20];

        for (int i = 0; i < m_useClasNum; i++) {
            if (fscanf(pFile, "%s", realValue)) {
                m_className.push_back((std::string)realValue);
                // m_className[i] = realValue;
                // std::cout << i << " : " << m_className[i] << '\n';

                while (fgetc(pFile) != '\n') ;
            } else {
                printf("there is something error\n");
                exit(-1);
            }
        }
    }

    fclose(pFile);
}

template<typename DTYPE> void ImageNetDataset<DTYPE>::CreateImageListOfEachClass() {
    // list file : 1st line - number of image, the others - image file name
    for (int classNum = 0; classNum < m_useClasNum; classNum++) {
        std::string filePath = m_rootPath + '/' + m_dataPath + '/' + m_className[classNum] + "/list.txt";  // check with printf
        const char *cstr     = filePath.c_str();

        FILE *pFile = NULL;
        pFile = fopen(cstr, "r");

        char realValue[100];
        int  numOfImageOfClass = 0;

        if (pFile == NULL) {
            printf("file open fail\n");
            exit(-1);
        } else {
            if (fscanf(pFile, "%s", realValue)) {  // first realValue is already readed above
                numOfImageOfClass = atoi(realValue);

                for (int imageNum = 0; imageNum < numOfImageOfClass; imageNum++) {
                    if (fscanf(pFile, "%s", realValue)) {
                        m_aImage.push_back((std::string)(m_className[classNum] + '/' + realValue));
                        m_aLable.push_back(classNum);
                        // std::cout << m_aImage.back() << " : " << m_aLable.back()  << " : " << m_aLable.size() << '\n';
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

    m_numOfImage = m_aImage.size();
    assert(m_numOfImage > 0);
    // std::cout << "m_numOfImage : " << m_numOfImage << '\n';
}

#ifdef __TURBOJPEG__

template<typename DTYPE> void ImageNetDataset<DTYPE>::AllocImageBuffer(int idx, ImageWrapper& imgWrp) {
    int   width, height;
    FILE *jpegFile         = NULL;
    unsigned char *jpegBuf = NULL;
    int pixelFormat        = TJPF_RGB;
    tjhandle tjInstance    = NULL;
    long     size;
    int inSubsamp, inColorspace;
    unsigned long jpegSize;

    // create file address
    std::string filePath = m_rootPath + '/' + m_dataPath + '/' + m_aImage[idx];  // check with printf
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

template<typename DTYPE> Tensor<DTYPE> *ImageNetDataset<DTYPE>::Image2Tensor(ImageWrapper& imgWrp, int doValueScaling) {
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

template<typename DTYPE> void ImageNetDataset<DTYPE>::Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling) {
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
            printf("file open fail\n");
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
