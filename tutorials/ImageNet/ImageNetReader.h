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

// #ifdef __TURBOJPEG__
#include <turbojpeg.h>
// #endif  // ifdef __TURBOJPEG__

#include "../../WICWIU_src/Tensor_utils.h"

#define NUMBER_OF_CLASS               1000
#define REAL_NUMBER_OF_CLASS          1000
#define NUMBER_OF_CHANNEL             3
#define LEGNTH_OF_WIDTH_AND_HEIGHT    224
#define CAPACITY_OF_PLANE             50176
#define CAPACITY_OF_IMAGE             150528
#define NUMBER_OF_THREAD              5

using namespace std;

template<typename DTYPE> class ImageNetDataReader {
private:
    string m_path             = "../../../../../../mnt/ssd/Data/ImageNet";
    string m_dirOfTrainImage  = "ILSVRC2012_img_train256";
    string m_dirOfTestImage   = "ILSVRC2012_img_val256";
    string m_classInformation = "synset_words.txt";

    /*Train image*/
    // list of Class
    string m_className[NUMBER_OF_CLASS];
    // for shuffle class index
    vector<int> m_shuffledList;
    // number of img of each class
    int m_aNumOfImageOfClass[NUMBER_OF_CLASS];
    // image set of each class
    string **m_aaImagesOfClass;

    /*Test image*/
    int m_numOfTestImage;
    int *m_classNumOfEachImage;
    string *m_listOfTestImage;

    // batch Tensor << before concatenate
    queue<Tensor<DTYPE> *> *m_aaSetOfImage;  // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabel;  // size : batch size

    queue<Tensor<DTYPE> *> *m_aaSetOfImageForConcatenate;  // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabelForConcatenate;  // size : batch size

    // Storage for preprocessed Tensor
    queue<Tensor<DTYPE> **> *m_aaQForData;  // buffer Size is independently define here

    // vector<int> m_shuffledListForImgNum;

    int m_batchSize;
    int m_recallnum;
    int m_bufferSize;

    int m_isTrain;

    pthread_t m_thread;

    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    int m_work;

    /*Data Preprocessing*/
    // for normalization
    int m_useNormalization;
    int m_isNormalizePerChannelWise;
    float *m_aMean;
    float *m_aStddev;

    /*Data Augmentation*/
    // Random_crop
    int m_useRandomCrop;
    int m_padding;
    vector<int> m_shuffledListForCrop;

    // Horizontal Flip
    int m_useRandomHorizontalFlip;
    vector<int> m_shuffledListForHorizontalFlip;

    // Vetical Flip
    int m_useRandomVerticalFlip;
    vector<int> m_shuffledListForVerticalFlip;

    // for multi thread
    queue<int> m_selectedClassNum;
    queue<int> m_selectedimgNum;

    sem_t m_fullForSelectedDataInformation;
    sem_t m_emptyForSelectedDataInformation;
    sem_t m_mutexForSelectedDataInformation;

    sem_t m_mutexForSingleImgTensor;

    sem_t m_mutexForSpaceChange;

    sem_t m_mutexForConcatenate;

private:
    int Alloc() {
        for (int i = 1; i < REAL_NUMBER_OF_CLASS; ++i) m_shuffledList.push_back(i);

        m_aaImagesOfClass = new string *[NUMBER_OF_CLASS];

        m_aaSetOfImage = new queue<Tensor<DTYPE> *>();  // Each tensor shows single image
        m_aaSetOfLabel = new queue<Tensor<DTYPE> *>();

        if (m_isTrain) {
            m_aaSetOfImageForConcatenate = new queue<Tensor<DTYPE> *>();  // Each tensor shows single image
            m_aaSetOfLabelForConcatenate = new queue<Tensor<DTYPE> *>();
        }

        m_aaQForData = new queue<Tensor<DTYPE> **>();  // Each tensor shows set of image which size is batchSize
        return TRUE;
    }

    void Delete() {
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
                    delete[] temp;
                    temp = NULL;
                }
            }
            delete m_aaQForData;
            m_aaQForData = NULL;
        }
    }

public:
    ImageNetDataReader(int batchSize, int bufferSize, int isTrain) {
        m_numOfTestImage = 0;

        m_batchSize  = batchSize;
        m_isTrain    = isTrain;
        m_recallnum  = 0;
        m_bufferSize = bufferSize;

        sem_init(&m_full,                            0, 0);
        sem_init(&m_empty,                           0, bufferSize);
        sem_init(&m_mutex,                           0, 1);

        sem_init(&m_fullForSelectedDataInformation,  0, 0);
        sem_init(&m_emptyForSelectedDataInformation, 0, batchSize);
        sem_init(&m_mutexForSelectedDataInformation, 0, 1);

        sem_init(&m_mutexForSingleImgTensor,         0, 1);

        sem_init(&m_mutexForSpaceChange,             0, 1);

        sem_init(&m_mutexForConcatenate,             0, 0);


        m_work = 1;

        Alloc();

        // prepare data what we need
        this->CheckClassList();
        this->CreateImageListOfEachClass();

        // start data preprocessing with above information
        // It works with thread
        // it will be end when receive "STOP" signal
    }

    virtual ~ImageNetDataReader() {
        Delete();
    }

    int StartProduce() {
        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, m_bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_work = 1;

        pthread_create(&m_thread, NULL, &ImageNetDataReader::ThreadFuncForDataPreprocess, (void *)this);

        return TRUE;
    }

    int StopProduce() {
        // some signal
        m_work = 0;
        // terminate every element
        sem_post(&m_empty);
        sem_post(&m_full);

        // thread join
        pthread_join(m_thread, NULL);

        if (m_aaQForData) {
            if (m_aaQForData->size() != 0) {
                int numOfTensor = m_aaQForData->size();

                for (int i = 0; i < numOfTensor; i++) {
                    Tensor<DTYPE> **temp = m_aaQForData->front();
                    m_aaQForData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete[] temp;
                    temp = NULL;
                }
            }
        }

        return TRUE;
    }

    int CheckClassList() {
        // for label one-hot vector
        // fix class index

        // mnt/ssd/Data/ImageNet/synset_words.txt
        string filePath  = m_path + "/" + m_classInformation;
        const char *cstr = filePath.c_str();

        // std::cout << filePath << '\n';

        FILE *pFile = NULL;

        // std::cout << filePath << '\n';
        // printf("%s\n", cstr);

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

        fclose(pFile);

        return TRUE;
    }

    int CreateImageListOfEachClass() {
        // for image preprocessing
        // when we have a list of image, we can shuffle the set of image data

        if (m_isTrain) {
            int count = 0;

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

                        count += m_aNumOfImageOfClass[classNum];

                        for (int imageNum = 0; imageNum < m_aNumOfImageOfClass[classNum]; imageNum++) {
                            if (fscanf(pFile, "%s", realValue)) {
                                listOfImage[imageNum] = realValue;
                                // std::cout << listOfImage[imageNum] << '\n';
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

                fclose(pFile);

                // std::cout << "test" << '\n';
            }
            std::cout << count << '\n';
        } else {
            for (int classNum = 0; classNum < NUMBER_OF_CLASS; classNum++) {
                string filePath  = m_path + '/' + m_dirOfTestImage + '/' + m_className[classNum] + "/list.txt"; // check with printf
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

                        int numOfImageOfClass = m_aNumOfImageOfClass[classNum];

                        m_numOfTestImage += m_aNumOfImageOfClass[classNum];
                        // std::cout << m_aNumOfImageOfClass[i] << '\n';
                        string *listOfImage = new string[m_aNumOfImageOfClass[classNum]];

                        for (int imageNum = 0; imageNum < numOfImageOfClass; imageNum++) {
                            if (fscanf(pFile, "%s", realValue)) {
                                listOfImage[imageNum] = realValue;
                                // std::cout << listOfImage[imageNum] << '\n';
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

                fclose(pFile);

                // std::cout << "test" << '\n';
            }


            // std::cout << "m_numOfTestImage : " << m_numOfTestImage << '\n';

            m_classNumOfEachImage = new int[m_numOfTestImage];
            m_listOfTestImage     = new string[m_numOfTestImage];
            int count = 0;

            for (int classNum = 0; classNum < NUMBER_OF_CLASS; classNum++) {
                int numOfImageOfClass = m_aNumOfImageOfClass[classNum];

                for (int subCount = 0; subCount < numOfImageOfClass; subCount++) {
                    m_classNumOfEachImage[count] = classNum;
                    m_listOfTestImage[count]     = m_aaImagesOfClass[classNum][subCount];

                    // std::cout << m_classNumOfEachImage[count] << " : " << m_listOfTestImage[count] << '\n';
                    count++;
                }
            }

            std::cout << count << '\n';
        }

        // std::cout << "test" << '\n';

        // class folder Address
        // set of image address of each folder
        return TRUE;
    }

    static void* ThreadFuncForDataPreprocess(void *arg) {
        ImageNetDataReader<DTYPE> *reader = (ImageNetDataReader<DTYPE> *)arg;

        reader->DataPreprocess();

        return NULL;
    }

    int DataPreprocess() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        // semaphore is used
        m_recallnum = 0;

        int classNum   = 0; // random class
        int imgNum     = 0;   // random image of above class
        string imgName = "\0";

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        if (m_isTrain) {
            this->ShuffleClassNum();

            pthread_t *setOfThread = (pthread_t *)malloc(sizeof(pthread_t) * NUMBER_OF_THREAD);
            pthread_t  setOfThreadForPushData2Buffer;

            // start thread
            for (size_t threadNum = 0; threadNum < NUMBER_OF_THREAD; threadNum++) {
                pthread_create(&(setOfThread[threadNum]), NULL, &ImageNetDataReader::ThreadFuncForData2Tensor, (void *)this);
            }

            pthread_create(&setOfThreadForPushData2Buffer, NULL, &ImageNetDataReader::ThreadFuncForPushData2Buffer, (void *)this);

            do {
                if (((m_recallnum + 1) * m_batchSize) > NUMBER_OF_CLASS) {
                    this->ShuffleClassNum();
                    m_recallnum = 0;
                }

                if (m_useRandomCrop) {
                    FillshuffledListForCrop();
                    srand(unsigned(time(0)));
                    random_shuffle(m_shuffledListForCrop.begin(), m_shuffledListForCrop.end(), ImageNetDataReader<DTYPE>::random_generator);
                }

                if (m_useRandomHorizontalFlip) {
                    FillshuffledListForHorizontalFlip();
                    srand(unsigned(time(0)));
                    random_shuffle(m_shuffledListForHorizontalFlip.begin(), m_shuffledListForHorizontalFlip.end(), ImageNetDataReader<DTYPE>::random_generator);
                }

                if (m_useRandomVerticalFlip) {
                    FillshuffledListForVerticalFlip();
                    srand(unsigned(time(0)));
                    random_shuffle(m_shuffledListForVerticalFlip.begin(), m_shuffledListForVerticalFlip.end(), ImageNetDataReader<DTYPE>::random_generator);
                }

                // std::cout << "test" << '\n';

                // std::cout << "m_recallnum : " << m_recallnum << '\n';

                for (int i = 0; i < m_batchSize; i++) {
                    classNum = m_shuffledList[i + m_recallnum * m_batchSize] % NUMBER_OF_CLASS;
                    // classNum = m_shuffledList[i + m_recallnum * m_batchSize];
                    // std::cout << classNum << ' ';
                    // std::cout << i + m_recallnum * m_batchSize << ' ';
                    // std::cout << classNum << ' ';
                    imgNum = rand() % m_aNumOfImageOfClass[classNum];  // random select from range(0, m_aNumOfImageOfClass[classNum])
                    // std::cout << m_aNumOfImageOfClass[classNum] << " : " << imgNum << '\n';
                    // m_aaSetOfImage->push(this->Image2Tensor(classNum, imgNum));
                    // m_aaSetOfLabel->push(this->Label2Tensor(classNum));
                    sem_wait(&m_emptyForSelectedDataInformation);
                    sem_wait(&m_mutexForSelectedDataInformation);

                    m_selectedClassNum.push(classNum);
                    m_selectedimgNum.push(imgNum);

                    sem_post(&m_mutexForSelectedDataInformation);
                    sem_post(&m_fullForSelectedDataInformation);
                }

                // std::cout << "test" << '\n';

                // sem_wait(&m_mutexForConcatenate);
                //
                // preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
                // preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);
                //
                // sem_wait(&m_empty);
                // sem_wait(&m_mutex);
                //
                // this->AddData2Buffer(preprocessedImages, preprocessedLabels);
                //
                // sem_post(&m_mutex);
                // sem_post(&m_full);

                // int empty_value = 0;
                // int full_value  = 0;
                //
                // sem_getvalue(&m_empty, &empty_value);
                // sem_getvalue(&m_full,  &full_value);
                //
                // printf("full : %d, empty : %d \n", full_value, empty_value);

                m_recallnum++;
            } while (m_work);

            for (size_t threadNum = 0; threadNum < NUMBER_OF_THREAD; threadNum++) {
                pthread_join(setOfThread[threadNum], NULL);
            }

            free(setOfThread);
        } else {
            do {
                if (((m_recallnum + 1) * m_batchSize) > m_numOfTestImage) {
                    m_recallnum = 0;
                }

                for (int i = 0; i < m_batchSize; i++) {
                    classNum = m_classNumOfEachImage[i + m_recallnum * m_batchSize] % NUMBER_OF_CLASS;
                    // std::cout << classNum << ' ';
                    // std::cout << i + m_recallnum * m_batchSize<< '\n';
                    // std::cout << classNum << ' ';
                    imgName = m_listOfTestImage[i + m_recallnum * m_batchSize];  // random select from range(0, m_aNumOfImageOfClass[classNum])
                    // std::cout << classNum << " : " << imgName << '\n';
                    m_aaSetOfImage->push(this->Image2Tensor(classNum, imgName));
                    m_aaSetOfLabel->push(this->Label2Tensor(classNum));
                }
                preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
                preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

                this->AddData2Buffer(preprocessedImages, preprocessedLabels);

                sem_post(&m_mutex);
                sem_post(&m_full);

                // int empty_value = 0;
                // int full_value  = 0;
                //
                // sem_getvalue(&m_empty, &empty_value);
                // sem_getvalue(&m_full,  &full_value);
                //
                // printf("full : %d, empty : %d \n", full_value, empty_value);

                m_recallnum++;
            } while (m_work);
        }

        return TRUE;
    }

    // threadfunction의 모체가 되는 함수
    static void* ThreadFuncForData2Tensor(void *arg) {
        ImageNetDataReader<DTYPE> *reader = (ImageNetDataReader<DTYPE> *)arg;

        reader->Data2Tensor();

        return NULL;
    }

    int Data2Tensor() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        // semaphore is used
        //
        int classNum = 0;  // random class
        int imgNum   = 0;     // random image of above class

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        if (m_isTrain) {
            do {
                sem_wait(&m_fullForSelectedDataInformation);
                sem_wait(&m_mutexForSelectedDataInformation);

                classNum = m_selectedClassNum.front();
                m_selectedClassNum.pop();
                imgNum = m_selectedimgNum.front();
                m_selectedimgNum.pop();

                sem_post(&m_mutexForSelectedDataInformation);
                sem_post(&m_emptyForSelectedDataInformation);

                // 안에서 막아줄 필요가 있다.
                preprocessedImages = this->Image2Tensor(classNum, imgNum);
                preprocessedLabels = this->Label2Tensor(classNum);

                sem_wait(&m_mutexForSingleImgTensor);

                m_aaSetOfImage->push(preprocessedImages);
                m_aaSetOfLabel->push(preprocessedLabels);

                if (m_aaSetOfImage->size() == m_batchSize) {
                    // std::cout << "m_aaSetOfImage->size()" << m_aaSetOfImage->size() << '\n';
                    sem_wait(&m_mutexForSpaceChange);

                    this->ChangeImageVectorSpace();
                    // std::cout << "m_aaSetOfImage->size()" << m_aaSetOfImage->size() << '\n';

                    sem_post(&m_mutexForConcatenate);
                }

                sem_post(&m_mutexForSingleImgTensor);
            } while (m_work);
        } else {
            std::cout << "their is something wrong mechanism on thread" << '\n';
            exit(-1);
        }

        return TRUE;
    }

    static void* ThreadFuncForPushData2Buffer(void *arg) {
        ImageNetDataReader<DTYPE> *reader = (ImageNetDataReader<DTYPE> *)arg;

        reader->PushData2Buffer();
        return NULL;
    }

    int PushData2Buffer() {
        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        do {
            sem_wait(&m_mutexForConcatenate);

            preprocessedImages = this->ConcatenateImage(m_aaSetOfImageForConcatenate);
            preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabelForConcatenate);

            sem_wait(&m_empty);
            sem_wait(&m_mutex);

            this->AddData2Buffer(preprocessedImages, preprocessedLabels);

            sem_post(&m_mutex);
            sem_post(&m_full);

            sem_post(&m_mutexForSpaceChange);
        } while (m_work);

        return TRUE;
    }

    int ChangeImageVectorSpace() {
        queue<Tensor<DTYPE> *> *tempForImageSpace = m_aaSetOfImage;
        queue<Tensor<DTYPE> *> *tempForLabelSpace = m_aaSetOfLabel;

        m_aaSetOfImage = m_aaSetOfImageForConcatenate;
        m_aaSetOfLabel = m_aaSetOfLabelForConcatenate;

        m_aaSetOfImageForConcatenate = tempForImageSpace;
        m_aaSetOfLabelForConcatenate = tempForLabelSpace;

        return TRUE;
    }

    static int random_generator(int upperbound) {
        return rand() % upperbound;
    }

    void ShuffleClassNum() {
        srand(unsigned(time(0)));
        random_shuffle(m_shuffledList.begin(), m_shuffledList.end(), ImageNetDataReader<DTYPE>::random_generator);
    }

    void Resize(int channel, int oldHeight, int oldWidth, unsigned char *oldData, int newHeight, int newWidth, unsigned char *newData) {
        unsigned char *dest = newData;

        for (int newy = 0; newy < newHeight; newy++) {
            int oldy = newy * oldHeight / newHeight;
            // if(oldy >= oldHeight)
            // oldy = oldHeight - 1;			// for safety
            unsigned char *srcLine = oldData + oldy * oldWidth * channel;

            for (int newx = 0; newx < newWidth; newx++) {
                int oldx = newx * oldWidth / newWidth;
                // if(oldx >= oldWidth)
                // oldx = oldWidth - 1;			// for safety
                unsigned char *src = srcLine + oldx * channel;

                for (int c = 0; c < channel; c++) *(dest++) = *(src++);
            }
        }
    }

    Tensor<DTYPE>* Image2Tensor(int classNum, int imgNum  /*Address of Image*/) {
        int   width, height;
        int   ch, ro, co;
        char *inFormat, *outFormat;
        FILE *jpegFile = NULL;
        unsigned char *imgBuf = NULL, *jpegBuf = NULL;
        int pixelFormat     = TJPF_RGB;
        tjhandle tjInstance = NULL;
        long     size;
        int inSubsamp, inColorspace;
        unsigned long jpegSize;
        // unsigned char * clone;
        int xOfImage = 0, yOfImage = 0;
        const int lengthLimit        = 224; // lengthLimit
        const int colorDim           = 3; // channel
        unsigned char *imgReshapeBuf = NULL;

        string classDir = m_className[classNum];
        // std::cout << classDir << '\n';

        string imgName = m_aaImagesOfClass[classNum][imgNum];
        // std::cout << "imgName : " << imgName << '\n';

        // create file address
        string filePath = m_path + '/' + m_dirOfTrainImage + '/' + classDir + '/' + imgName;  // check with printf

        const char *cstr = filePath.c_str();

        // string _FILENAME     = "temp/" + imgName;
        // const char *FILENAME = _FILENAME.c_str();

        // std::cout << "filePath : " << filePath << '\n';

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, colorDim, lengthLimit, lengthLimit);

        // Load image (no throw and catch)
        /* Read the JPEG file into memory. */
        jpegFile = fopen(cstr, "rb");

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

        if ((width < lengthLimit) || (height < lengthLimit)) {
            int newHeight = 0, newWidth = 0;

            if (width < height) {
                newHeight     = height * (float)lengthLimit / width;
                newWidth      = lengthLimit;
                imgReshapeBuf = new unsigned char[colorDim * newHeight * newWidth];
            } else {
                newHeight     = lengthLimit;
                newWidth      = width * (float)lengthLimit / height;
                imgReshapeBuf = new unsigned char[colorDim * newHeight * newWidth];
            }
            Resize(colorDim, height, width, imgBuf, newHeight, newWidth, imgReshapeBuf);

            width  = newWidth;
            height = newHeight;
        }

        // convert image to tensor
        //// if (width != lengthLimit) xOfImage = random_generator(width - lengthLimit);
        if (width != lengthLimit) xOfImage = (width - lengthLimit) / 2;

        // printf("width - lengthLimit %d - %d\n", width, lengthLimit);

        //// if (height != lengthLimit) yOfImage = random_generator(height - lengthLimit);
        if (height != lengthLimit) yOfImage = (height - lengthLimit) / 2;

        // printf("height - lengthLimit %d - %d\n", height, lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        // should be modularized
        for (ro = 0; ro < lengthLimit; ro++) {
            for (co = 0; co < lengthLimit; co++) {
                for (ch = 0; ch < colorDim; ch++) {
                    if (imgReshapeBuf == NULL) (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)] = imgBuf[(yOfImage + ro) * width * colorDim + (xOfImage + co) * colorDim + ch] / 255.0;
                    else (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)] = imgReshapeBuf[(yOfImage + ro) * width * colorDim + (xOfImage + co) * colorDim + ch] / 255.0;
                }
            }
        }

        if (m_useNormalization) {
            temp = Normalization(temp);
        }

        if (m_useRandomCrop) {
            temp = Padding(temp);
            temp = RandomCrop(temp);
        }

        if (m_useRandomHorizontalFlip) {
            if (m_shuffledListForHorizontalFlip.back()) {
                temp = HorizontalFlip(temp);
            }
            m_shuffledListForHorizontalFlip.pop_back();
        }

        if (m_useRandomVerticalFlip) {
            if (m_shuffledListForVerticalFlip.back()) temp = VerticalFlip(temp);
            m_shuffledListForVerticalFlip.pop_back();
        }

        // ImageNetDataReader::Tensor2Image(temp, FILENAME, colorDim, lengthLimit, lengthLimit);

        tjFree(imgBuf);
        delete[] imgReshapeBuf;

        temp->ReShape(1, 1, 1, 1, colorDim * lengthLimit * lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        return temp;
    }

    Tensor<DTYPE>* Image2Tensor(int classNum, string imgName  /*Address of Image*/) {
        int   width, height;
        int   ch, ro, co;
        char *inFormat, *outFormat;
        FILE *jpegFile = NULL;
        unsigned char *imgBuf = NULL, *jpegBuf = NULL;
        int pixelFormat     = TJPF_RGB;
        tjhandle tjInstance = NULL;
        long     size;
        int inSubsamp, inColorspace;
        unsigned long jpegSize;
        // unsigned char * clone;
        int xOfImage = 0, yOfImage = 0;
        const int lengthLimit        = 224; // lengthLimit
        const int colorDim           = 3; // channel
        unsigned char *imgReshapeBuf = NULL;

        string classDir = m_className[classNum];
        // std::cout << classDir << '\n';

        // create file address
        string filePath = m_path + '/' + m_dirOfTestImage + '/' + classDir + '/' + imgName;  // check with printf

        const char *cstr = filePath.c_str();

        // string _FILENAME     = "temp/" + imgName;
        // const char *FILENAME = _FILENAME.c_str();

        // std::cout << "filePath : " << filePath << '\n';

        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, colorDim, lengthLimit, lengthLimit);

        // Load image (no throw and catch)
        /* Read the JPEG file into memory. */
        jpegFile = fopen(cstr, "rb");

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

        if ((width < lengthLimit) || (height < lengthLimit)) {
            int newHeight = 0, newWidth = 0;

            if (width < height) {
                newHeight     = height * (float)lengthLimit / width;
                newWidth      = lengthLimit;
                imgReshapeBuf = new unsigned char[colorDim * newHeight * newWidth];
            } else {
                newHeight     = lengthLimit;
                newWidth      = width * (float)lengthLimit / height;
                imgReshapeBuf = new unsigned char[colorDim * newHeight * newWidth];
            }
            Resize(colorDim, height, width, imgBuf, newHeight, newWidth, imgReshapeBuf);

            width  = newWidth;
            height = newHeight;
        }

        // convert image to tensor
        // if (width != lengthLimit) xOfImage = random_generator(width - lengthLimit);
        if (width != lengthLimit) xOfImage = (width - lengthLimit) / 2;

        // printf("width - lengthLimit %d - %d\n", width, lengthLimit);

        // if (height != lengthLimit) yOfImage = random_generator(height - lengthLimit);
        if (height != lengthLimit) yOfImage = (height - lengthLimit) / 2;

        // printf("height - lengthLimit %d - %d\n", height, lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        // should be modularized
        for (ro = 0; ro < lengthLimit; ro++) {
            for (co = 0; co < lengthLimit; co++) {
                for (ch = 0; ch < colorDim; ch++) {
                    if (imgReshapeBuf == NULL) (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)] = imgBuf[(yOfImage + ro) * width * colorDim + (xOfImage + co) * colorDim + ch] / 255.0;
                    else (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)] = imgReshapeBuf[(yOfImage + ro) * width * colorDim + (xOfImage + co) * colorDim + ch] / 255.0;
                }
            }
        }

        // ImageNetDataReader::Tensor2Image(temp, FILENAME, colorDim, lengthLimit, lengthLimit);

        tjFree(imgBuf);
        delete[] imgReshapeBuf;

        temp->ReShape(1, 1, 1, 1, colorDim * lengthLimit * lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        return temp;
    }

    void Tensor2Image(Tensor<DTYPE> *temp, const char *FILENAME, int colorDim, int height, int width) {
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
                    imgBuf[ro * width * colorDim + co * colorDim + ch] = (*temp)[Index5D(temp->GetShape(), 0, 0, ch, ro, co)] * 255.0;
                }
            }
        }

        tjInstance = tjInitCompress();
        tjCompress2(tjInstance, imgBuf, width, 0, height, pixelFormat,
                    &jpegBuf, &jpegSize,  /*outSubsamp =*/ TJSAMP_444,  /*outQual =*/ 100,  /*flags =*/ 0);
        tjDestroy(tjInstance);
        tjInstance = NULL;
        delete imgBuf;

        // std::cout << FILENAME << '\n';

        if (!(jpegFile = fopen(FILENAME, "wb"))) {
            printf("file open fail\n");
            exit(-1);
        }

        fwrite(jpegBuf, jpegSize, 1, jpegFile);
        fclose(jpegFile); jpegFile = NULL;
        tjFree(jpegBuf); jpegBuf   = NULL;
    }

    // #endif  // ifdef __TURBOJPEG__

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

    /////////////////////////////////////////////////////////////////////////////Data Preprocessing
    int UseNormalization(int isNormalizePerChannelWise, float *mean, float *stddev) {
        m_useNormalization          = TRUE;
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

    Tensor<DTYPE>* Normalization(Tensor<DTYPE> *image) {
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

    /////////////////////////////////////////////////////////////////////////////Data Augmentation
    int UseRandomCrop(int padding) {
        m_useRandomCrop = TRUE;
        m_padding       = padding;
        return TRUE;
    }

    int FillshuffledListForCrop() {
        int limitOfCropPos      = (2 * m_padding) + 1;
        int twoTimesOfBatchSize = m_batchSize * 2;  // for x and y

        for (int cntNum = 0; cntNum < twoTimesOfBatchSize; cntNum++) {
            m_shuffledListForCrop.push_back(cntNum % limitOfCropPos);
        }
        return TRUE;
    }

    Tensor<DTYPE>* Padding(Tensor<DTYPE> *srcImage) {
        int numOfChannel     = NUMBER_OF_CHANNEL;
        int heightOfImg      = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int widthOfImg       = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int planeSizeOfImage = LEGNTH_OF_WIDTH_AND_HEIGHT * LEGNTH_OF_WIDTH_AND_HEIGHT;

        int padding           = m_padding;
        int heightOfTensor    = heightOfImg + (2 * padding);
        int widthOfTensor     = widthOfImg + (2 * padding);
        int planeSizeOfTensor = heightOfTensor * widthOfTensor;

        Tensor<DTYPE> *padedImg = Tensor<DTYPE>::Zeros(1, 1, numOfChannel, heightOfTensor, widthOfTensor);

        int srcIdx = 0;
        int idx    = 0;

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int rowNum = 0; rowNum < heightOfImg; rowNum++) {
                for (int colNum = 0; colNum < widthOfImg; colNum++) {
                    srcIdx           = channelNum * planeSizeOfImage + rowNum * widthOfImg + colNum;
                    idx              = channelNum * planeSizeOfTensor + (padding + rowNum) * widthOfTensor + (padding + colNum);
                    (*padedImg)[idx] = (*srcImage)[srcIdx];
                }
            }
        }

        delete srcImage;

        return padedImg;
    }

    Tensor<DTYPE>* RandomCrop(Tensor<DTYPE> *srcImage) {
        int numOfChannel           = NUMBER_OF_CHANNEL;
        int heightOfSrcImg         = LEGNTH_OF_WIDTH_AND_HEIGHT + (2 * m_padding);
        int widthOfSrcImg          = LEGNTH_OF_WIDTH_AND_HEIGHT + (2 * m_padding);
        int srcImageSizePerChannel = heightOfSrcImg * widthOfSrcImg;
        int heightOfImg            = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int widthOfImg             = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int imageSizePerChannel    = heightOfImg * widthOfImg;

        Tensor<DTYPE> *cropedImg = Tensor<DTYPE>::Zeros(1, 1, numOfChannel, heightOfImg, widthOfImg);

        int startPosW = m_shuffledListForCrop.back();
        m_shuffledListForCrop.pop_back();
        int startPosH = m_shuffledListForCrop.back();
        m_shuffledListForCrop.pop_back();

        int srcIdx = 0;
        int idx    = 0;

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int rowNum = 0; rowNum < heightOfImg; rowNum++) {
                for (int colNum = 0; colNum < widthOfImg; colNum++) {
                    srcIdx            = channelNum * srcImageSizePerChannel + (startPosH + rowNum) * widthOfSrcImg + (startPosW + colNum);
                    idx               = channelNum * imageSizePerChannel + rowNum * widthOfImg + colNum;
                    (*cropedImg)[idx] = (*srcImage)[srcIdx];
                }
            }
        }

        delete srcImage;

        return cropedImg;
    }

    int UseRandomHorizontalFlip() {
        m_useRandomHorizontalFlip = TRUE;
        return TRUE;
    }

    int FillshuffledListForHorizontalFlip() {
        for (int cntNum = 0; cntNum < m_batchSize; cntNum++) {
            m_shuffledListForHorizontalFlip.push_back(cntNum % 2);
        }
        return TRUE;
    }

    Tensor<DTYPE>* HorizontalFlip(Tensor<DTYPE> *image) {
        int numOfChannel        = 3;
        int imageSizePerChannel = LEGNTH_OF_WIDTH_AND_HEIGHT * LEGNTH_OF_WIDTH_AND_HEIGHT;
        int rowSizePerPlane     = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int colSizePerPlane     = LEGNTH_OF_WIDTH_AND_HEIGHT;
        int halfColSizePerPlane = colSizePerPlane / 2;

        DTYPE temp          = 0;
        int   idx           = 0;
        int   idxOfOpposite = 0;

        for (int channelNum = 0; channelNum < numOfChannel; channelNum++) {
            for (int rowNum = 0; rowNum < rowSizePerPlane; rowNum++) {
                for (int colNum = 0; colNum < halfColSizePerPlane; colNum++) {
                    idx                     = channelNum * imageSizePerChannel + rowNum * colSizePerPlane + colNum;
                    idxOfOpposite           = channelNum * imageSizePerChannel + rowNum * colSizePerPlane + (colSizePerPlane - colNum - 1);
                    temp                    = (*image)[idx];
                    (*image)[idx]           = (*image)[idxOfOpposite];
                    (*image)[idxOfOpposite] = temp;
                }
            }
        }

        return image;
    }

    int UseRandomVerticalFlip() {
        m_useRandomVerticalFlip = TRUE;
        return TRUE;
    }

    int FillshuffledListForVerticalFlip() {
        for (int cntNum = 0; cntNum < m_batchSize; cntNum++) {
            m_shuffledListForVerticalFlip.push_back(cntNum % 2);
        }
        return TRUE;
    }

    Tensor<DTYPE>* VerticalFlip(Tensor<DTYPE> *image) {
        return NULL;
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

        // int empty_value = 0;
        // int full_value  = 0;
        //
        // sem_getvalue(&m_empty, &empty_value);
        // sem_getvalue(&m_full,  &full_value);
        //
        // printf("full : %d, empty : %d \n", full_value, empty_value);

        return result;
    }
};
