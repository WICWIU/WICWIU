#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <map>

#include "../../WICWIU_src/Tensor.h"

#define PATH                  "~/mnt/ssd/Data/ImageNet/"
#define DIR_OF_TRAIN_IMAGE    "ILSVRC2012_img_train256/"
#define DIR_OF_TEST_IMAGE     "ILSVRC2012_img_val256/"

#define NUMBER_OF_CLASS       1000

using namespace std;

template<typename DTYPE> class ImageNetDataReader {
private:
    // list of Class
    string m_className[NUMBER_OF_CLASS];
    // for shuffle class index
    std::vector<int> *m_aShuffledList;
    // number of img of each class
    int m_aNumOfImageOfClass[NUMBER_OF_CLASS];
    // image set of each class
    string **m_aaImagesOfClass;
    // batch Tensor << before concatenate
    vector<Tensor<DTYPE> *> *m_aaSetOfImage;  // size : batch size
    vector<Tensor<DTYPE> *> *m_aaSetOfLabel;  // size : batch size

    // Storage for preprocessed Tensor
    vector<Tensor<DTYPE> *> *m_aaCircularQ;  // buffer Size is independently define here

    int m_batchSize;
    int m_bufferSize;
    int m_recallnum;

    int m_isTrain;

private:
    int Alloc() {
        int numOfClass[NUMBER_OF_CLASS] = { 0 };

        for (int i = 0; i < NUMBER_OF_CLASS; i++) numOfClass[i] = i;
        m_aShuffledList = new vector<int>(numOfClass, numOfClass + NUMBER_OF_CLASS);

        m_aNumOfImageOfClass = new int[NUMBER_OF_CLASS];
        m_aaImagesOfClass    = new string *[NUMBER_OF_CLASS];

        m_aaSetOfImage = new vector<Tensor<DTYPE> *>();  // Each tensor shows single image
        m_aaSetOfLabel = new vector<Tensor<DTYPE> *>();

        m_aaCircularQ = new vector<Tensor<DTYPE> *>();  // Each tensor shows set of image which size is batchSize
        return TRUE;
    }

    void Delete() {
        if (m_aShuffledList) {
            delete m_aShuffledList;
            m_aShuffledList = NULL;
        }

        if (m_aNumOfImageOfClass) {
            delete m_aNumOfImageOfClass;
            m_aNumOfImageOfClass = NULL;
        }

        if (m_aaImagesOfClass) {
            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                if (m_aaImagesOfClass[i]) {
                    delete m_aaImagesOfClass[i];
                    m_aaImagesOfClass[i] = NULL;
                }
            }
            delete m_aaImagesOfClass;
            m_aaImagesOfClass = NULL;
        }

        if (m_aaSetOfImage) {
            if (m_aaSetOfImage.size()) {
                int numOfTensor = m_aaSetOfImage.size();

                for (int i = 0; i < numOfTensor; i++) {
                    if ((*m_aaSetOfImage)[i]) {
                        delete (*m_aaSetOfImage)[i];
                        (*m_aaSetOfImage)[i] = NULL;
                    }
                }
            }
            delete m_aaSetOfImage;
            m_aaSetOfImage = NULL;
        }

        if (m_aaSetOfLabel) {
            if (m_aaSetOfLabel.size()) {
                int numOfTensor = m_aaSetOfLabel.size();

                for (int i = 0; i < numOfTensor; i++) {
                    if ((*m_aaSetOfLabel)[i]) {
                        delete (*m_aaSetOfLabel)[i];
                        (*m_aaSetOfLabel)[i] = NULL;
                    }
                }
            }
            delete m_aaSetOfLabel;
            m_aaSetOfLabel = NULL;
        }

        if (m_aaCircularQ) {
            if (m_aaCircularQ.size()) {
                int numOfTensor = m_aaCircularQ.size();

                for (int i = 0; i < numOfTensor; i++) {
                    if ((*m_aaCircularQ)[i]) {
                        delete (*m_aaCircularQ)[i];
                        (*m_aaCircularQ)[i] = NULL;
                    }
                }
            }
            delete m_aaCircularQ;
            m_aaCircularQ = NULL;
        }
    }

public:
    ImageNetDataReader(int batchSize, int bufferSize, int isTrain) {
        m_batchSize  = batchSize;
        m_bufferSize = bufferSize;
        m_isTrain    = isTrain;
        m_recallnum  = 0;

        Alloc();

        // prepare data what we need
        this->CheckClassList();
        this->CreateImageListOfEachClass();

        // start data preprocessing with above information
        // it will be end when recieve "STOP" signal
        this->DataPreprocess();
    }

    virtual ~ImageNetDataReader() {
        Delete();
    }

    int DataPreprocess() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        // semapore is used

        int classNum = 0;  // random class
        int imgNum   = 0;     // random image of above class

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        while (TRUE  /*with semapore*/) {
            // if(/*some mechanism*/ ) this->Shuffle();

            for (int i = 0; i < m_batchSize; i++) {
                classNum = i + m_recallnum;
                imgNum   = rand() % m_aNumOfImageOfClass[classNum]; // random select from range(0, m_aNumOfImageOfClass[classNum])
                m_aaSetOfImage->push_back(this->Image2Tensor(classNum, imgNum));
                m_aaSetOfLabel->push_back(this->Image2Label(classNum));
            }

            this->ConcatenateImage(m_aaSetOfImage);
            this->ConcatenateLabel(m_aaSetOfImage);
        }


        return TRUE;
    }

    int CheckClassList() {
        // for label one-hot vector
        // fix class index
        ///mnt/ssd/Data/ImageNet/synset_words.txt - class

        for (int i = 0; i < NUMBER_OF_CLASS; i++) {
            // m_className[i] = synset_words.txt[i] : first name;
        }

        return TRUE;
    }

    int CreateImageListOfEachClass() {
        // for image preprocessing
        // when we have a list of image, we can shuffle the set of image data

        if (m_isTrain == TRUE) {
            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                // string temp = m_className[i];
                // temp = PATH + DIR_OF_TRAIN_IMAGE + temp // check with printf
                // m_aNumOfImageOfClass[i] = numOfImageOfTargeClass;
                // string * listOfImage = new string[numOfImageOfTargeClass]
                // for (int i = 0; i < numOfImageOfTargeClass; i++) {/*listOfImage[i] = img_name[i]*/}
            }
        } else {
            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                // string temp = m_className[i];
                // temp = PATH + DIR_OF_TEST_IMAGE + temp // check with printf
                // m_aNumOfImageOfClass[i] = numOfImageOfTargeClass;
                // string * listOfImage = new string[numOfImageOfTargeClass]
                // for (int i = 0; i < numOfImageOfTargeClass; i++) {/*listOfImage[i] = img_name[i]*/}
            }
        }

        // class folder Address
        // set of image address of each folder
        return TRUE;
    }

    Tensor<DTYPE>* Image2Tensor(int classNum, int imgNum  /*Address of Image*/) {
        string classDir = m_className[classNum];
        string imgName  = m_aaImagesOfClass[classNum][imgNum];

        // string imgPath = PATH + DIR_OF_TRAIN_IMAGE + classDir + imgName; // check with print address

        // load img with imgName
        // train
        return NULL;
    }

    Tensor<DTYPE>* Label2Tensor(int classNum  /*Address of Label*/) {
        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, NUMBER_OF_CLASS);
        (*temp)[classNum] = (DTYPE)1;
        return temp;
    }

    Tensor<DTYPE>* ConcatenateImage(  /*array of ImageTensor*/) {
        // develop
        // Do not consider efficiency
        return NULL;
    }

    Tensor<DTYPE>* ConcatenateLabel(  /*array of LabelTensor*/) {
        // develop
        // library 사용
        return NULL;
    }

    int AddData2Buffer(Tensor<DTYPE> **imageAndLabel) {
        return TRUE;
    }

    Tensor<DTYPE>** GetDataFromBuffer() {
        // circular queue
        return NULL;
    }

    int StopDataPreprocess() {
        // some signal with semapore
        return TRUE;
    }
};
