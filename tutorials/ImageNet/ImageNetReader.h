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
#include <turbojpeg.h>

#include "../../WICWIU_src/Tensor_utils.h"

// #define _throw(action, message) { \
//   printf("ERROR in line %d while %s:\n%s\n", __LINE__, action, message); \
//   retval = -1;  goto bailout; \
// }
// #define _throwtj(action)  _throw(action, tjGetErrorStr2(tjInstance))
// #define _throwunix(action)  _throw(action, strerror(errno))

#define NUMBER_OF_CLASS    1000

using namespace std;

template<typename DTYPE> class ImageNetDataReader {
private:
    string m_path             = "../../../../../../mnt/ssd/Data/ImageNet";
    string m_dirOfTrainImage  = "ILSVRC2012_img_train256";
    string m_dirOfTestImage   = "ILSVRC2012_img_val256/";
    string m_classInformation = "synset_words.txt";

    /*Training image*/
    // list of Class
    string m_className[NUMBER_OF_CLASS];
    // for shuffle class index
    vector<int> m_shuffledList;
    // number of img of each class
    int m_aNumOfImageOfClass[NUMBER_OF_CLASS];
    // image set of each class
    string **m_aaImagesOfClass;

    /*Testing image*/
    int m_numOfTestImage;
    int *m_classNumOfEachImage;
    string *m_listOfTestImage;

    // batch Tensor << before concatenate
    queue<Tensor<DTYPE> *> *m_aaSetOfImage;  // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabel;  // size : batch size

    // Storage for preprocessed Tensor
    queue<Tensor<DTYPE> **> *m_aaQForData;  // buffer Size is independently define here

    int m_batchSize;
    int m_recallnum;
    int m_bufferSize;

    int m_isTrain;

    pthread_t m_thread;

    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    int m_work;

private:
    int Alloc() {
        for (int i = 1; i < NUMBER_OF_CLASS; ++i) m_shuffledList.push_back(i);

        m_aaImagesOfClass = new string *[NUMBER_OF_CLASS];

        m_aaSetOfImage = new queue<Tensor<DTYPE> *>();  // Each tensor shows single image
        m_aaSetOfLabel = new queue<Tensor<DTYPE> *>();

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

        m_batchSize = batchSize;
        m_isTrain   = isTrain;
        m_recallnum = 0;
        m_bufferSize = bufferSize;

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
        // it will be end when receive "STOP" signal

    }

    virtual ~ImageNetDataReader() {
        Delete();
    }

    int StartProduce(){
        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, m_bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_work = 1;

        pthread_create(&m_thread, NULL, &ImageNetDataReader::ThreadFunc, (void *)this);

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

        fclose(pFile);

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

                fclose(pFile);

            }
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

                fclose(pFile);
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
        m_recallnum = 0;

        int classNum   = 0; // random class
        int imgNum     = 0;   // random image of above class
        string imgName = "\0";

        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;

        if (m_isTrain) {
            this->ShuffleClassNum();

            do {
                if (((m_recallnum + 1) * m_batchSize) > NUMBER_OF_CLASS) {
                    this->ShuffleClassNum();
                    m_recallnum = 0;
                }

                // std::cout << "m_recallnum : " << m_recallnum << '\n';

                for (int i = 0; i < m_batchSize; i++) {
                    classNum = m_shuffledList[i + m_recallnum * m_batchSize];
                    // std::cout << classNum << ' ';
                    // std::cout << i + m_recallnum * m_batchSize << ' ';
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

                // int empty_value = 0;
                // int full_value  = 0;
                //
                // sem_getvalue(&m_empty, &empty_value);
                // sem_getvalue(&m_full,  &full_value);
                //
                // printf("full : %d, empty : %d \n", full_value, empty_value);

                m_recallnum++;
            } while (m_work);
        } else {
            do {
                if(((m_recallnum + 1) * m_batchSize) > m_numOfTestImage){
                    std::cout << "resume" << '\n';
                    m_recallnum = 0;
                }


                for (int i = 0; i < m_batchSize; i++) {
                    classNum = m_classNumOfEachImage[i + m_recallnum * m_batchSize];
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
        if (width != lengthLimit) xOfImage = random_generator(width - lengthLimit);

        // printf("width - lengthLimit %d - %d\n", width, lengthLimit);

        if (height != lengthLimit) yOfImage = random_generator(height - lengthLimit);

        // printf("height - lengthLimit %d - %d\n", height, lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        // should be modularized
        for (ch = 0; ch < colorDim; ch++) {
            for (ro = yOfImage; ro < lengthLimit; ro++) {
                for (co = xOfImage; co < lengthLimit; co++) {
                    if (imgReshapeBuf == NULL) (*temp)[Index3D(temp->GetShape(), ch, ro, co)] = imgBuf[ro * lengthLimit * colorDim + co * colorDim + ch] / 255.0;
                    else (*temp)[Index3D(temp->GetShape(), ch, ro, co)] = imgReshapeBuf[ro * lengthLimit * colorDim + co * colorDim + ch] / 255.0;
                }
            }
        }

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
        if (width != lengthLimit) xOfImage = random_generator(width - lengthLimit);

        // printf("width - lengthLimit %d - %d\n", width, lengthLimit);

        if (height != lengthLimit) yOfImage = random_generator(height - lengthLimit);

        // printf("height - lengthLimit %d - %d\n", height, lengthLimit);

        // std::cout << temp->GetShape() << '\n';

        // should be modularized
        for (ch = 0; ch < colorDim; ch++) {
            for (ro = yOfImage; ro < lengthLimit; ro++) {
                for (co = xOfImage; co < lengthLimit; co++) {
                    if (imgReshapeBuf == NULL) (*temp)[Index3D(temp->GetShape(), ch, ro, co)] = imgBuf[ro * lengthLimit * colorDim + co * colorDim + ch] / 255.0;
                    else (*temp)[Index3D(temp->GetShape(), ch, ro, co)] = imgReshapeBuf[ro * lengthLimit * colorDim + co * colorDim + ch] / 255.0;
                }
            }
        }

        tjFree(imgBuf);
        delete[] imgReshapeBuf;

        temp->ReShape(1, 1, 1, 1, colorDim * lengthLimit * lengthLimit);

        // std::cout << temp->GetShape() << '\n';

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
