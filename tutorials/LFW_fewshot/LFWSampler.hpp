#ifndef SAMPLER_H_
#define SAMPLER_H_    value

#define __TURBOJPEG__

#ifdef __TURBOJPEG__
# include <turbojpeg.h>
#endif

#include "../../WICWIU_src/DataLoader.hpp"
#include "LFWDataset.hpp"

template<typename DTYPE>
class LFWSampler : public DataLoader<DTYPE>{
private:
    /* data */
    int m_numOfClass;

public:
    LFWSampler(int numOfClass, Dataset<DTYPE> *dataset, int batchSize = 1, int useShuffle = FALSE, int numOfWorker = 1, int dropLast = TRUE);
    virtual ~LFWSampler();

    virtual void MakeAllOfIndex(std::vector<int> *pAllOfIndex);

    virtual void DataPreprocess();
    void Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling);
};

template<typename DTYPE> LFWSampler<DTYPE>::LFWSampler(int numOfClass, Dataset<DTYPE> *dataset, int batchSize, int useShuffle, int numOfWorker, int dropLast)
    : DataLoader<DTYPE>(dataset, batchSize, useShuffle, numOfWorker, dropLast) {
    m_numOfClass = numOfClass;
}

template<typename DTYPE> LFWSampler<DTYPE>::~LFWSampler() {}

template<typename DTYPE> void LFWSampler<DTYPE>::MakeAllOfIndex(std::vector<int> *pAllOfIndex)
{
    pAllOfIndex->resize(0);
    LFWDataset<DTYPE> *pDataset = (LFWDataset<DTYPE>*)this->GetDataset();
    for (int i = 0; i < pDataset->GetLength(); i++){
        if(pDataset->GetSampleCount(pDataset->GetLabel(i)) > 1)
            pAllOfIndex->push_back(i);
    }

#ifdef  __DEBUG__
        LogMessageF("anchor_sample.txt", TRUE, "%d anchor classes\n", pAllOfIndex->size());
        for(int i = 0; i < pAllOfIndex->size(); i++)
            LogMessageF("anchor_sample.txt", FALSE, "%d\t%d\t%d\n", (*pAllOfIndex)[i], pDataset->GetLabel((*pAllOfIndex)[i]), pDataset->GetSampleCount(pDataset->GetLabel((*pAllOfIndex)[i])));

        // MyPause(__FUNCTION__);
#endif  //  __DEBUG__
}

template<typename DTYPE> void LFWSampler<DTYPE>::DataPreprocess() {
    // for thread
    // doing all of thing befor push global buffer
    // arrange everything for worker

    int num						 = 0;
    int m_batchSize              = this->GetBatchSize();

    int numOfAnchorSample        = m_batchSize / 3;
    int numOfPosSample           = numOfAnchorSample;
    int numOfNegSample           = numOfAnchorSample;
    int m_numOfEachDatasetMember = this->GetNumOfEachDatasetMember();

    Dataset<DTYPE> *m_pDataset = this->GetDataset();

    std::queue<Tensor<DTYPE> *> *localBuffer       = new std::queue<Tensor<DTYPE> *>[m_numOfEachDatasetMember];
    std::vector<int> *setOfIdx                     = NULL;
    int label                                      = 0;
    std::vector<Tensor<DTYPE> *> *data             = NULL;
    std::vector<Tensor<DTYPE> *> *preprocessedData = NULL;

    std::queue<int> setOfLabel;
    std::cout << "SAMPLER worker" << '\n';
    std::cout << "numOfAnchorSample : " << numOfAnchorSample << '\n';
    std::srand(time(NULL));

    while (this->GetWorkingSignal()) {
        // get information from IdxBuffer
        setOfIdx = this->GetIdxSetFromIdxBuffer();
        
        for(int i = 0; i < numOfAnchorSample; i++){
            int anchorIdx = (*setOfIdx)[i];
            label = m_pDataset->GetLabel(anchorIdx);

            // push anchor sample
            data = m_pDataset->GetData(anchorIdx);
            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
                // this -> Tensor2Image("anc_img.jpeg", (*data)[0], TRUE);
                // if(j == 1) std::cout << (*data)[j] << '\n';
            }
            // std::cin >> label;

            delete data;
            data = NULL;


            // push positive sample
            int posIdx = 0;
            data = m_pDataset->GetDataOfPositiveLabel(anchorIdx, &posIdx);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
                // this -> Tensor2Image("pos_img.jpeg", (*data)[0], TRUE);
                // if(j == 1) std::cout << (*data)[j] << '\n';
            }

            delete data;
            data = NULL;

            // push negative sample
            int negIdx = 0;
            data = m_pDataset->GetDataOfNegativeLabel(anchorIdx, &negIdx);

            for (int j = 0; j < m_numOfEachDatasetMember; j++) {
                // Chech the type of Data for determine doing preprocessing (IMAGE)
                // if true do data Preprocessing
                // push data into local buffer
                localBuffer[j].push((*data)[j]);
                // this -> Tensor2Image("neg_img.jpeg", (*data)[0], TRUE);
                // if(j == 1) std::cout << (*data)[j] << '\n';
            }

            delete data;
            data = NULL;


#ifdef  __DEBUG__
            // LFWDataset<float> *lfw = (LFWDataset<float>*)m_pDataset;
            // printf("anchor sample path: %s\n", lfw->GetImagePath(anchorIdx).c_str());
            // printf("positive sample path: %s\n", lfw->GetImagePath(posIdx).c_str());
            // printf("negative sample path: %s\n", lfw->GetImagePath(negIdx).c_str());
            // printf("Press Enter to continue...");
            // getchar();
#endif  //  __DEBUG__
        }
        // std::cin >> num;
        // delete set of idx vector

        delete setOfIdx;
        setOfIdx = NULL;

        preprocessedData = new std::vector<Tensor<DTYPE> *>(m_numOfEachDatasetMember, NULL);  // do not deallocate in this function!

        for (int k = 0; k < m_numOfEachDatasetMember; k++) {
            // concatenate each localbuffer
            // push preprocessedData vector
            (*preprocessedData)[k] = this->Concatenate(localBuffer[k]);
        }

        // push preprocessedData into Global buffer
        this->Push2GlobalBuffer(preprocessedData);
        preprocessedData = NULL;
        // std::cin >> num;
    }

    delete[] localBuffer;
}

template<typename DTYPE> void LFWSampler<DTYPE>::Tensor2Image(std::string filename, Tensor<DTYPE> *imgTensor, int doValuerScaling) {
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
#endif  // ifndef SAMPLER_H_
