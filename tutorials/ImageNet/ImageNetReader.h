#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include "../../WICWIU_src/Tensor_utils.h"

#define NUMBER_OF_CLASS    1000

using namespace std;

template<typename DTYPE> class ImageNetDataReader {
private:
    string m_path             = "../../../../../../mnt/ssd/Data/ImageNet";
    string m_dirOfTrainImage  = "ILSVRC2012_img_train256";
    string m_dirOfTestImage   = "ILSVRC2012_img_val256/";
    string m_classInformation = "synset_words.txt";

    // list of Class
    string m_className[NUMBER_OF_CLASS];
    // for shuffle class index
    std::vector<int> *m_aShuffledList;
    // number of img of each class
    int m_aNumOfImageOfClass[NUMBER_OF_CLASS];
    // image set of each class
    string **m_aaImagesOfClass;
    // batch Tensor << before concatenate
    queue<Tensor<DTYPE> *> *m_aaSetOfImage;  // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabel;  // size : batch size

    // Storage for preprocessed Tensor
    queue<Tensor<DTYPE> **> *m_aaQForData;  // buffer Size is independently define here

    int m_batchSize;
    int m_recallnum;

    int m_isTrain;

    pthread_t m_thread;

    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    int m_work;

private:
    int Alloc() {
        int numOfClass[NUMBER_OF_CLASS] = { 0 };

        for (int i = 0; i < NUMBER_OF_CLASS; i++) numOfClass[i] = i;
        m_aShuffledList = new vector<int>(numOfClass, numOfClass + NUMBER_OF_CLASS);

        m_aaImagesOfClass = new string *[NUMBER_OF_CLASS];

        m_aaSetOfImage = new queue<Tensor<DTYPE> *>();  // Each tensor shows single image
        m_aaSetOfLabel = new queue<Tensor<DTYPE> *>();

        m_aaQForData = new queue<Tensor<DTYPE> **>();  // Each tensor shows set of image which size is batchSize
        return TRUE;
    }

    void Delete() {
        if (m_aShuffledList) {
            delete m_aShuffledList;
            m_aShuffledList = NULL;
        }

        if (m_aaImagesOfClass) {
            // We cannot dealloc this part, is it charactor of string value?
            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                if (m_aaImagesOfClass[i]) {
                    // std::cout << m_aaImagesOfClass[i] << '\n';
                    delete[] m_aaImagesOfClass[i];
                    m_aaImagesOfClass[i] = NULL;
                }
            }

            delete m_aaImagesOfClass;
            m_aaImagesOfClass = NULL;
        }

        if (m_aaSetOfImage) {
            if (m_aaSetOfImage->size() != 0) {
                int numOfTensor = m_aaSetOfImage->size();

                for (int i = 0; i < numOfTensor; i++) {
                    delete m_aaSetOfImage->front();
                    m_aaSetOfImage->front() = NULL;
                    m_aaSetOfImage->pop();
                }
            }
            delete m_aaSetOfImage;
            m_aaSetOfImage = NULL;
        }

        if (m_aaSetOfLabel) {
            if (m_aaSetOfLabel->size() != 0) {
                int numOfTensor = m_aaSetOfLabel->size();

                for (int i = 0; i < numOfTensor; i++) {
                    delete m_aaSetOfLabel->front();
                    m_aaSetOfLabel->front() = NULL;
                    m_aaSetOfLabel->pop();
                }
            }
            delete m_aaSetOfLabel;
            m_aaSetOfLabel = NULL;
        }

        if (m_aaQForData) {
            if (m_aaQForData->size() != 0) {
                int numOfTensor = m_aaQForData->size();

                for (int i = 0; i < numOfTensor; i++) {
                    Tensor<DTYPE> **temp = m_aaQForData->front();
                    m_aaQForData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete temp;
                    temp = NULL;
                }
            }
            delete m_aaQForData;
            m_aaQForData = NULL;
        }
    }

public:
    ImageNetDataReader(int batchSize, int bufferSize, int isTrain) {
        m_batchSize = batchSize;
        m_isTrain   = isTrain;
        m_recallnum = 0;

        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_work = 1;

        Alloc();

        // prepare data what we need
        this->CheckClassList();
        this->CreateImageListOfEachClass();

        // start data preprocessing with above information
        // It works with thread
        // it will be end when recieve "STOP" signal

        pthread_create(&m_thread, NULL, &ImageNetDataReader::ThreadFunc, (void *)this);
    }

    virtual ~ImageNetDataReader() {
        Delete();
    }

    int CheckClassList() {
        // for label one-hot vector
        // fix class index

        // mnt/ssd/Data/ImageNet/synset_words.txt
        string filePath  = m_path + '/' + m_classInformation;
        const char *cstr = filePath.c_str();

        // std::cout << filePath << '\n';

        FILE *pFile = NULL;

        pFile = fopen(cstr, "r");

        if (pFile == NULL) {
            printf("file open fail\n");
            exit(-1);
        } else {
            char realValue[20];

            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                if (fscanf(pFile, "%s", realValue)) {
                    m_className[i] = realValue;
                    // std::cout << m_className[i] << '\n';

                    while (fgetc(pFile) != '\n') ;
                } else {
                    printf("there is something error\n");
                    exit(-1);
                }
            }
        }

        return TRUE;
    }

    int CreateImageListOfEachClass() {
        // for image preprocessing
        // when we have a list of image, we can shuffle the set of image data

        if (m_isTrain) {
            for (int classNum = 0; classNum < NUMBER_OF_CLASS; classNum++) {
                string filePath  = m_path + '/' + m_dirOfTrainImage + '/' + m_className[classNum] + "/list.txt"; // check with printf
                const char *cstr = filePath.c_str();

                // list file : 1st line - number of image, the others - image file name
                FILE *pFile = NULL;
                pFile = fopen(cstr, "r");

                if (pFile == NULL) {
                    printf("file open fail\n");
                    exit(-1);
                } else {
                    char realValue[100];

                    if (fscanf(pFile, "%s", realValue)) {
                        m_aNumOfImageOfClass[classNum] = atoi(realValue);
                        // std::cout << m_aNumOfImageOfClass[i] << '\n';
                        string *listOfImage = new string[m_aNumOfImageOfClass[classNum]];

                        for (int imageNum = 0; imageNum < m_aNumOfImageOfClass[classNum]; imageNum++) {
                            if (fscanf(pFile, "%s", realValue)) {
                                listOfImage[imageNum] = realValue;
                                // std::cout << listOfImage[i] << '\n';
                            } else {
                                printf("there is something error\n");
                                exit(-1);
                            }
                        }

                        m_aaImagesOfClass[classNum] = listOfImage;
                    } else {
                        printf("there is something error\n");
                        exit(-1);
                    }
                }
            }
        } else {
            for (int i = 0; i < NUMBER_OF_CLASS; i++) {
                // string temp = m_className[i];
                // temp = m_path + m_dirOfTestImage + temp // check with printf
                // m_aNumOfImageOfClass[i] = numOfImageOfTargeClass;
                // string * listOfImage = new string[numOfImageOfTargeClass]
                // for (int i = 0; i < numOfImageOfTargeClass; i++) {/*listOfImage[i] = img_name[i]*/}
            }
        }

        // class folder Address
        // set of image address of each folder
        return TRUE;
    }

    static void* ThreadFunc(void *arg) {
        ImageNetDataReader<DTYPE> *reader = (ImageNetDataReader<DTYPE> *)arg;
        reader->DataPreprocess();

        return NULL;
    }

    int DataPreprocess() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        // semaphore is used

        int classNum = 0;  // random class
        int imgNum   = 0;     // random image of above class

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        if (m_isTrain) {
            do {
                if (((m_recallnum + 1) * m_batchSize - 1) > NUMBER_OF_CLASS) m_recallnum = 0  /* this->Shuffle()*/;

                std::cout << "m_recallnum : " << m_recallnum << '\n';

                for (int i = 0; i < m_batchSize; i++) {
                    classNum = (*m_aShuffledList)[i + m_recallnum * m_batchSize];
                    // std::cout << i + m_recallnum * m_batchSize << '\n';
                    // std::cout << classNum << ' ';
                    imgNum = rand() % m_aNumOfImageOfClass[classNum];  // random select from range(0, m_aNumOfImageOfClass[classNum])
                    // std::cout << m_aNumOfImageOfClass[classNum] << " : " << imgNum << '\n';
                    m_aaSetOfImage->push(this->Image2Tensor(classNum, imgNum));
                    m_aaSetOfLabel->push(this->Label2Tensor(classNum));
                }
                preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
                preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

                this->AddData2Buffer(preprocessedImages, preprocessedLabels);

                sem_post(&m_mutex);
                sem_post(&m_full);

                m_recallnum++;
            } while (m_work);
        } else {
            do {
                std::cout << "not now! it will be prepared" << '\n';
                exit(-1);
            } while (m_work  /*with semaphore*/);
        }

        return TRUE;
    }

    Tensor<DTYPE>* Image2Tensor(int classNum, int imgNum  /*Address of Image*/) {
        string classDir = m_className[classNum];
        // std::cout << classDir << '\n';
        string imgName = m_aaImagesOfClass[classNum][imgNum];
        // std::cout << "imgName : " << imgName << '\n';

        string filePath  = m_path + '/' + m_dirOfTrainImage + '/' + classDir + '/' + imgName; // check with printf

        // std::cout << "filePath : " << filePath << '\n';

        int width    = 224; // column
        int height   = 224; // row
        int colorDim = 3;  // channel

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, colorDim * height * width);
        // Tensor<DTYPE> *temp = Tensor<DTYPE>::Constants(1, 1, 1, 1, colorDim * height * width, classNum);

        // Load image
        // convert image to tensor

        return temp;
    }

    Tensor<DTYPE>* Label2Tensor(int classNum  /*Address of Label*/) {
        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, NUMBER_OF_CLASS);
        (*temp)[classNum] = (DTYPE)1;
        return temp;
    }

    Tensor<DTYPE>* ConcatenateImage(queue<Tensor<DTYPE> *> *setOfImage) {
        // Do not consider efficiency
        int singleImageSize = setOfImage->front()->GetCapacity();

        Tensor<DTYPE> *result      = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, singleImageSize);
        Tensor<DTYPE> *singleImage = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++) {
            singleImage = setOfImage->front();
            setOfImage->pop();

            for (int idxOfImage = 0; idxOfImage < singleImageSize; idxOfImage++) {
                int idxOfResult = batchNum * singleImageSize + idxOfImage;
                (*result)[idxOfResult] = (*singleImage)[idxOfImage];
            }

            // dealloc single image
            delete singleImage;
            singleImage = NULL;
        }

        // setOfImage->clear();

        return result;
    }

    Tensor<DTYPE>* ConcatenateLabel(queue<Tensor<DTYPE> *> *setOfLabel) {
        // Do not consider efficiency
        Tensor<DTYPE> *result      = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, NUMBER_OF_CLASS);
        Tensor<DTYPE> *singleLabel = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++) {
            singleLabel = setOfLabel->front();
            setOfLabel->pop();

            for (int idxOfLabel = 0; idxOfLabel < NUMBER_OF_CLASS; idxOfLabel++) {
                int idxOfResult = batchNum * NUMBER_OF_CLASS + idxOfLabel;
                (*result)[idxOfResult] = (*singleLabel)[idxOfLabel];
            }

            // dealloc single image
            delete singleLabel;
            singleLabel = NULL;
        }

        // setOfLabel->clear();

        return result;
    }

    int AddData2Buffer(Tensor<DTYPE> *setOfImage, Tensor<DTYPE> *setOfLabel) {
        Tensor<DTYPE> **result = new Tensor<DTYPE> *[2];

        result[0] = setOfImage;
        result[1] = setOfLabel;

        m_aaQForData->push(result);

        return TRUE;
    }

    Tensor<DTYPE>** GetDataFromBuffer() {
        sem_wait(&m_full);
        sem_wait(&m_mutex);

        Tensor<DTYPE> **result = m_aaQForData->front();
        m_aaQForData->pop();

        sem_post(&m_mutex);
        sem_post(&m_empty);

        return result;
    }

    int StopDataPreprocess() {
        // some signal
        m_work = 0;
        // terminate every element
        sem_post(&m_empty);
        sem_post(&m_full);

        // thread join
        pthread_join(m_thread, NULL);

        std::cout << "Data Reader Thread is terminated!" << '\n';

        return TRUE;
    }
};
