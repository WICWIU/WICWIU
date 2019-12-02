#include "../../../WICWIU_src/DataLoader.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>

// #ifdef __TURBOJPEG__
#include <turbojpeg.h>
// #endif  // ifdef __TURBOJPEG__


#define CAPACITY_OF_PLANE             4096
#define CAPACITY_OF_IMAGE             12288
#define NUMBER_OF_IMAGE               202599
#define LENGTH_64                     64
#define NUMBER_OF_CHANNEL             3
#define LEGNTH_OF_WIDTH_AND_HEIGHT    64

using namespace std;

template<typename DTYPE> class CelebADataset : public Dataset<DTYPE>{
private:
    /* data */
    string m_imagePath;
    vector<string> m_imageName;
    float *m_aMean;
    float *m_aStddev;
    int m_isNormalizePerChannelWise;

public:
    CelebADataset(string pImagePath);
    ~CelebADataset();
    void Alloc();
    void Dealloc();
    void initImageNames();
    std::vector<Tensor<DTYPE> *>* GetData(int idx);
    int GetLength(); // all of data Length
    void Tensor2Image(Tensor<DTYPE> *temp, const char *FILENAME, int colorDim, int batch, int height, int width);
    int UseNormalization(int isNormalizePerChannelWise, float *mean, float *stddev);
    Tensor<DTYPE>* Normalization(Tensor<DTYPE> *image);
};

template<typename DTYPE> CelebADataset<DTYPE>::CelebADataset(string pImagePath){
    m_imagePath = pImagePath;

    this->Alloc();
    this->initImageNames();

}
template<typename DTYPE> CelebADataset<DTYPE>::~CelebADataset(){

    Dealloc();
}

template<typename DTYPE> void CelebADataset<DTYPE>::Alloc(){
}

template<typename DTYPE> void CelebADataset<DTYPE>::Dealloc(){
}

template<typename DTYPE> void CelebADataset<DTYPE>::initImageNames(){
    string temp = "";
        for(int i=1; i<=NUMBER_OF_IMAGE; i++){
            if(i / 10 == 0){
            temp = "00000" + to_string(i) + ".jpg";
            }
            else if(i / 100 == 0){
                temp = "0000" + to_string(i) + ".jpg";
            }
            else if(i / 1000 == 0){
                temp = "000" + to_string(i) + ".jpg";
            }
            else if(i / 10000 == 0){
                temp = "00" + to_string(i) + ".jpg";
            }
            else if(i / 100000 == 0){
                temp = "0" + to_string(i) + ".jpg";
            }
            else{
                temp = to_string(i) + ".jpg";
            }
            m_imageName.push_back(temp);
        }
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *>* CelebADataset<DTYPE>::GetData(int idx){
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    string filePath = m_imagePath + "/" + m_imageName[idx];


    int width, height;
    FILE *jpegFile = NULL;
    long     size;
    unsigned char *imgBuf = NULL, *jpegBuf = NULL;
    tjhandle tjInstance = NULL;
    unsigned long jpegSize;
    int pixelFormat     = TJPF_RGB;
    int inSubsamp, inColorspace;

    const char *cstr = filePath.c_str();
    Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 3, LENGTH_64, LENGTH_64); // multi channel일 떄 col에 체널수도 곱해줘야함.

    jpegFile = fopen(cstr, "rb");
    if(jpegFile == NULL){
        std::cout << "There is No " << cstr << std::endl;
    }
    fseek(jpegFile, 0, SEEK_END);
    size = ftell(jpegFile);
    fseek(jpegFile, 0, SEEK_SET);

    jpegSize = (unsigned long)size;
    jpegBuf  = (unsigned char *)tjAlloc(jpegSize);

    if (fread(jpegBuf, jpegSize, 1, jpegFile) < 1) exit(-1);
        fclose(jpegFile); jpegFile = NULL;

    tjInstance = tjInitDecompress();
    tjDecompressHeader3(tjInstance, jpegBuf, jpegSize, &width, &height, &inSubsamp, &inColorspace);

    imgBuf = (unsigned char *)tjAlloc(width * height * tjPixelSize[pixelFormat]);
    tjDecompress2(tjInstance, jpegBuf, jpegSize, imgBuf, width, 0, height, pixelFormat, 0);
    tjFree(jpegBuf); jpegBuf          = NULL;
    tjDestroy(tjInstance); tjInstance = NULL;


    for (int ro = 0; ro < height; ro++) {
        for (int co = 0; co < width; co++) {
            for (int ch = 0; ch < NUMBER_OF_CHANNEL; ch++) {
                // //for multi channel
                 (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)]
                     = imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL + ch] / 255.0 * 2 - 1;

                // for gray scale (-1~1)
                // (*temp)[Index5D(temp->GetShape(), 0, 0, 0, ro, co)]
                //     = (imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL] / 255.0 * 2 - 1
                //     + imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL + 1] / 255.0 * 2 - 1
                //     + imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL + 2] / 255.0 * 2 - 1) / 3;

                // (*temp)[Index5D(temp->GetShape(), 0, 0, 0, ro, co)]
                //      = (imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL] / 255.0
                //      + imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL + 1] / 255.0
                //      + imgBuf[ro * width * NUMBER_OF_CHANNEL + co * NUMBER_OF_CHANNEL + 2] / 255.0) / 3;
            }
        }
    }
    // temp = Normalization(temp);
    tjFree(imgBuf);

    temp->ReShape(1, 1, 1, 1, 3* LENGTH_64 * LENGTH_64); // multi channel일 떄 col에 체널수도 곱해줘야함.

    result->push_back(temp);
    return result;
}

template<typename DTYPE>  int CelebADataset<DTYPE>:: UseNormalization(int isNormalizePerChannelWise, float *mean, float *stddev) {
    //m_useNormalization          = TRUE;
    m_isNormalizePerChannelWise = isNormalizePerChannelWise;

    if (m_isNormalizePerChannelWise) {
            m_aMean   = new float[NUMBER_OF_CHANNEL];
            m_aStddev = new float[NUMBER_OF_CHANNEL];

            for (int channelNum = 0; channelNum < NUMBER_OF_CHANNEL; channelNum++) {
                    m_aMean[channelNum]   = mean[channelNum];
                    m_aStddev[channelNum] = stddev[channelNum];
            }
    } else {
            m_aMean   = new float[CAPACITY_OF_IMAGE];
            m_aStddev = new float[CAPACITY_OF_IMAGE];

            for (int elementNum = 0; elementNum < CAPACITY_OF_IMAGE; elementNum++) {
                    m_aMean[elementNum]   = mean[elementNum];
                    m_aStddev[elementNum] = stddev[elementNum];
            }
    }

    return TRUE;
}

template<typename DTYPE> Tensor<DTYPE>* CelebADataset<DTYPE>:: Normalization(Tensor<DTYPE> *image) {
    int numOfChannel     = NUMBER_OF_CHANNEL;
    int heightOfImg      = LEGNTH_OF_WIDTH_AND_HEIGHT;
    int widthOfImg       = LEGNTH_OF_WIDTH_AND_HEIGHT;
    int planeSizeOfImage = LEGNTH_OF_WIDTH_AND_HEIGHT * LEGNTH_OF_WIDTH_AND_HEIGHT;

    int heightOfTensor    = heightOfImg;
    int widthOfTensor     = widthOfImg;
    int planeSizeOfTensor = heightOfTensor * widthOfTensor;

    if (m_isNormalizePerChannelWise) {
            int idx = 0;

            for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
                    for (int heightIdx = 0; heightIdx < heightOfImg; heightIdx++) {
                            for (int widthIdx = 0; widthIdx < widthOfImg; widthIdx++) {
                                    idx            = channelNum * planeSizeOfTensor + (heightIdx) * widthOfTensor + (widthIdx);
                                    (*image)[idx] -= m_aMean[channelNum];
                                    (*image)[idx] /= m_aStddev[channelNum];
                            }
                    }
            }
    } else {
            int idx                = 0;
            int idxOfMeanAdnStddev = 0;

            for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
                    for (int heightIdx = 0; heightIdx < heightOfImg; heightIdx++) {
                            for (int widthIdx = 0; widthIdx < widthOfImg; widthIdx++) {
                                    idx                = channelNum * planeSizeOfTensor + (heightIdx) * widthOfTensor + (widthIdx);
                                    idxOfMeanAdnStddev = channelNum * planeSizeOfImage + heightIdx * widthOfImg + widthIdx;
                                    (*image)[idx]     -= m_aMean[channelNum];
                                    (*image)[idx]     /= m_aStddev[channelNum];
                            }
                    }
            }
    }

    return image;
}

template<typename DTYPE> int CelebADataset<DTYPE>:: GetLength(){
    return NUMBER_OF_IMAGE;
}

template<typename DTYPE>void Tensor2Image(Tensor<DTYPE> *temp, const char *FILENAME, int colorDim, int height, int width) {
    unsigned char *imgBuf   = new unsigned char[colorDim * height * width];
    int pixelFormat         = TJPF_RGB;
    unsigned char *jpegBuf  = NULL;  /* Dynamically allocate the JPEG buffer */
    unsigned long  jpegSize = 0;
    FILE *jpegFile          = NULL;
    tjhandle tjInstance     = NULL;

    if (!temp) {
        printf("Invalid Tensor pointer");
        exit(-1);
    }

    for (int ro = 0; ro < height; ro++) {
        for (int co = 0; co < width; co++) {
            for (int ch = 0; ch < colorDim; ch++) {
                //for multi channel
                imgBuf[ro * width * colorDim + co * colorDim + ch] = ((*temp)[Index5D(temp->GetShape(), 0, 0, 0, 0, ch * height * width + ro * width + co)] + 1) * 255.0 / 2;

                //for gray scale (-1~1)
                // imgBuf[ro * width * colorDim + co * colorDim + ch] = ((*temp)[Index5D(temp->GetShape(), 0, 0, 0, 0, ro * width + co)] + 1) * 255.0 / 2;

                // imgBuf[ro * width * colorDim + co * colorDim + ch] = ((*temp)[Index5D(temp->GetShape(), 0, 0, 0, 0, ro * width + co)]) * 255.0;

            }


        }
    }

    tjInstance = tjInitCompress();
    tjCompress2(tjInstance, imgBuf, width, 0, height, pixelFormat,
                &jpegBuf, &jpegSize,  /*outSubsamp =*/ TJSAMP_444,  /*outQual =*/ 100,  /*flags =*/ 0);
    tjDestroy(tjInstance);
    tjInstance = NULL;
    delete imgBuf;

    if (!(jpegFile = fopen(FILENAME, "wb"))) {
        printf("file open fail\n");
        exit(-1);
    }

    fwrite(jpegBuf, jpegSize, 1, jpegFile);
    fclose(jpegFile); jpegFile = NULL;
    tjFree(jpegBuf); jpegBuf   = NULL;
}
