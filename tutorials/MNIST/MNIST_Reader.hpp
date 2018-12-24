#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include "../../WICWIU_src/Tensor.hpp"

#define DIMOFMNISTIMAGE     784
#define DIMOFMNISTLABEL     10

#define NUMOFTESTDATA       10000
#define NUMOFTRAINDATA      60000

#define TEST_IMAGE_FILE     "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_FILE     "data/t10k-labels-idx1-ubyte"
#define TRAIN_IMAGE_FILE    "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_FILE    "data/train-labels-idx1-ubyte"

using namespace std;

enum OPTION {
    TESTING,
    TESTIMAGE,
    TESTLABEL,
    TRAINING,
    TRAINIMAGE,
    TRAINLABEL,
    DEFAULT
};
int random_generator(int upperbound) {
    return rand() % upperbound;
}

template<typename DTYPE>
class MNISTDataSet {
private:
    DTYPE **m_aaTestImage;
    DTYPE **m_aaTestLabel;
    DTYPE **m_aaTrainImage;
    DTYPE **m_aaTrainLabel;

    // µû·Î ÇØÁ¦
    Tensor<DTYPE> *m_aTestImageFeed;
    Tensor<DTYPE> *m_aTestLabelFeed;
    Tensor<DTYPE> *m_aTrainImageFeed;
    Tensor<DTYPE> *m_aTrainLabelFeed;

    queue<Tensor<DTYPE> **> *m_aaQForData;  // buffer Size is independently define here

    pthread_t m_thread;
    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    vector<int> *m_ShuffledListTest;
    vector<int> *m_ShuffledListTrain;

    int m_RecallNumOfTest;
    int m_RecallNumOfTrain;

    int m_AsTrainData;

    int m_batchsize;

    int m_work;

    int m_isGPUData;

public:
    MNISTDataSet(int pAsTrainData = TRUE, int bufferSize = 10, int batchSize = 100, int isGPUData = -1) {
        m_aaTestImage  = NULL;
        m_aaTestLabel  = NULL;
        m_aaTrainImage = NULL;
        m_aaTrainLabel = NULL;

        m_aTestImageFeed  = NULL;
        m_aTestLabelFeed  = NULL;
        m_aTrainImageFeed = NULL;
        m_aTrainLabelFeed = NULL;

        m_ShuffledListTest  = NULL;
        m_ShuffledListTrain = NULL;

        m_RecallNumOfTest  = 0;
        m_RecallNumOfTrain = 0;

        m_AsTrainData = pAsTrainData;

        sem_init(&m_full,  0, 0);
        sem_init(&m_empty, 0, bufferSize);
        sem_init(&m_mutex, 0, 1);

        m_batchsize = batchSize;
        m_work      = TRUE;
        m_isGPUData = isGPUData;

        Alloc();
    }

    virtual ~MNISTDataSet() {
        Delete();
    }

    void Alloc() {
        int aNumTest[NUMOFTESTDATA]   = { 0 };
        int aNumTrain[NUMOFTRAINDATA] = { 0 };

        for (int i = 0; i < NUMOFTESTDATA; i++) aNumTest[i] = i;

        for (int i = 0; i < NUMOFTRAINDATA; i++) aNumTrain[i] = i;

        m_ShuffledListTest  = new vector<int>(aNumTest, aNumTest + NUMOFTESTDATA);
        m_ShuffledListTrain = new vector<int>(aNumTrain, aNumTrain + NUMOFTRAINDATA);

        m_aaQForData = new queue<Tensor<DTYPE> **> ();
    }

    void Delete() {
        for (int i = 0; i < NUMOFTESTDATA; i++) {
            delete[] m_aaTestImage[i];
            delete[] m_aaTestLabel[i];
        }
        delete m_aaTestImage;
        delete m_aaTestLabel;

        for (int i = 0; i < NUMOFTRAINDATA; i++) {
            delete[] m_aaTrainImage[i];
            delete[] m_aaTrainLabel[i];
        }
        delete m_aaTrainImage;
        delete m_aaTrainLabel;


        // Feed Tensor ÇØÁ¦ ¸øÇÔ
        //
        // delete m_aTestImageFeed;
        // delete m_aTestLabelFeed;
        // delete m_aTrainImageFeed;
        // delete m_aTrainLabelFeed;
    }

    void StartProduce() {
        pthread_create(&m_thread, NULL, &MNISTDataSet::ThreadFuncForDataPreprocess, (void *)this);
    }

    static void* ThreadFuncForDataPreprocess(void *arg) {
        MNISTDataSet<DTYPE> *reader = (MNISTDataSet<DTYPE> *)arg;

        reader->DataPreprocess();

        return NULL;
    }

    int DataPreprocess() {
        if (m_AsTrainData) {
            do {
                this->CreateTrainDataPair(m_batchsize);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

#ifdef __CUDNN__
                if (m_isGPUData != -1) {
                    m_aTrainImageFeed->SetDeviceGPU(m_isGPUData);
                    m_aTrainLabelFeed->SetDeviceGPU(m_isGPUData);
                }
#endif  // __CUDNN__
                this->AddData2Buffer(m_aTrainImageFeed, m_aTrainLabelFeed);

                sem_post(&m_mutex);
                sem_post(&m_full);

                m_aTrainImageFeed = NULL;
                m_aTrainLabelFeed = NULL;
            } while (m_work);
        } else {
            do {
                this->CreateTestDataPair(m_batchsize);

                sem_wait(&m_empty);
                sem_wait(&m_mutex);

#ifdef __CUDNN__
                if (m_isGPUData != -1) {
                    m_aTestImageFeed->SetDeviceGPU(m_isGPUData);
                    m_aTestLabelFeed->SetDeviceGPU(m_isGPUData);
                }
#endif  // __CUDNN__

                this->AddData2Buffer(m_aTestImageFeed, m_aTestLabelFeed);

                sem_post(&m_mutex);
                sem_post(&m_full);

                m_aTestImageFeed = NULL;
                m_aTestLabelFeed = NULL;
            } while (m_work);
        }

        return TRUE;
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

    void CreateTestDataPair(int pBatchSize) {
        if (pBatchSize * (m_RecallNumOfTest + 1) > NUMOFTESTDATA) {
            m_RecallNumOfTest = 0;
        }

        int numOfTestData = NUMOFTESTDATA;
        int recallNum     = m_RecallNumOfTest;
        int startPoint    = 0;
        int curPoint      = 0;

        vector<int> *shuffledList = m_ShuffledListTest;

        m_aTestImageFeed = new Tensor<DTYPE>(1, pBatchSize, 1, 1, DIMOFMNISTIMAGE);
        m_aTestLabelFeed = new Tensor<DTYPE>(1, pBatchSize, 1, 1, DIMOFMNISTLABEL);

        startPoint = (recallNum * pBatchSize) % numOfTestData;

        for (int ba = 0; ba < pBatchSize; ba++) {
            curPoint = (*shuffledList)[startPoint + ba];

            for (int co = 0; co < DIMOFMNISTIMAGE; co++) {
                (*m_aTestImageFeed)[Index5D(m_aTestImageFeed->GetShape(), 0, ba, 0, 0, co)] = m_aaTestImage[curPoint][co];
            }

            for (int co = 0; co < DIMOFMNISTLABEL; co++) {
                (*m_aTestLabelFeed)[Index5D(m_aTestLabelFeed->GetShape(), 0, ba, 0, 0, co)] = 0;
            }

            (*m_aTestLabelFeed)[Index5D(m_aTestLabelFeed->GetShape(), 0, ba, 0, 0, (int)m_aaTestLabel[curPoint][0])] = 1.0;
        }
        m_RecallNumOfTest++;
    }

    void CreateTrainDataPair(int pBatchSize) {
        if (pBatchSize * (m_RecallNumOfTrain + 1) > NUMOFTRAINDATA) {
            m_RecallNumOfTrain = 0;
            ShuffleDataPair(TRAINING, pBatchSize);
        }

        int numOfTrainData = NUMOFTRAINDATA;
        int recallNum      = m_RecallNumOfTrain;
        int startPoint     = 0;
        int curPoint       = 0;

        vector<int> *shuffledList = m_ShuffledListTrain;

        m_aTrainImageFeed = new Tensor<DTYPE>(1, pBatchSize, 1, 1, DIMOFMNISTIMAGE);
        m_aTrainLabelFeed = new Tensor<DTYPE>(1, pBatchSize, 1, 1, DIMOFMNISTLABEL);

        startPoint = (recallNum * pBatchSize) % numOfTrainData;

        for (int ba = 0; ba < pBatchSize; ba++) {
            curPoint = (*shuffledList)[startPoint + ba];

            for (int co = 0; co < DIMOFMNISTIMAGE; co++) {
                (*m_aTrainImageFeed)[Index5D(m_aTrainImageFeed->GetShape(), 0, ba, 0, 0, co)] = m_aaTrainImage[curPoint][co];
            }

            for (int co = 0; co < DIMOFMNISTLABEL; co++) {
                (*m_aTrainLabelFeed)[Index5D(m_aTrainLabelFeed->GetShape(), 0, ba, 0, 0, co)] = 0;
            }
            (*m_aTrainLabelFeed)[Index5D(m_aTrainLabelFeed->GetShape(), 0, ba, 0, 0, (int)m_aaTrainLabel[curPoint][0])] = 1.0;
        }
        m_RecallNumOfTrain++;
    }

    void ShuffleDataPair(OPTION pOption, int pBatchSize) {
        srand(unsigned(time(0)));
        vector<int> *shuffledList = NULL;

        if (pOption == TESTING) shuffledList = m_ShuffledListTest;
        else if (pOption == TRAINING) shuffledList = m_ShuffledListTrain;
        else {
            cout << "invalid OPTION!" << '\n';
            exit(0);
        }
        random_shuffle(shuffledList->begin(), shuffledList->end(), random_generator);
    }

    void SetTestImage(DTYPE **pTestImage) {
        m_aaTestImage = pTestImage;
    }

    void SetTestLabel(DTYPE **pTestLabel) {
        m_aaTestLabel = pTestLabel;
    }

    void SetTrainImage(DTYPE **pTrainImage) {
        m_aaTrainImage = pTrainImage;
    }

    void SetTrainLabel(DTYPE **pTrainLabel) {
        m_aaTrainLabel = pTrainLabel;
    }

    Tensor<DTYPE>* GetTestFeedImage() {
        return m_aTestImageFeed;
    }

    Tensor<DTYPE>* GetTrainFeedImage() {
        return m_aTrainImageFeed;
    }

    Tensor<DTYPE>* GetTestFeedLabel() {
        return m_aTestLabelFeed;
    }

    Tensor<DTYPE>* GetTrainFeedLabel() {
        return m_aTrainLabelFeed;
    }
};

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

template<typename DTYPE> void IMAGE_Reader(string DATAPATH, DTYPE **pImage) {
    ifstream fin;

    if (DATAPATH == TEST_IMAGE_FILE) {
        fin.open(TEST_IMAGE_FILE, ios_base::binary);
    } else if (DATAPATH == TRAIN_IMAGE_FILE) {
        fin.open(TRAIN_IMAGE_FILE, ios_base::binary);
    }

    if (fin.is_open()) {
        int magicNumber = 0;
        int numOfImage  = 0;
        int n_rows      = 0;
        int n_cols      = 0;

        fin.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin.read((char *)&numOfImage,  sizeof(numOfImage));
        numOfImage = ReverseInt(numOfImage);

        fin.read((char *)&n_rows,      sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        fin.read((char *)&n_cols,      sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int dimOfImage = n_rows * n_cols;

        for (int i = 0; i < numOfImage; ++i) {
            pImage[i] = new DTYPE[dimOfImage];

            for (int d = 0; d < dimOfImage; ++d) {
                unsigned char data = 0;
                fin.read((char *)&data, sizeof(data));
                pImage[i][d] = (DTYPE)data / 255.0;
            }
        }
    }
    fin.close();
}

template<typename DTYPE>
void LABEL_Reader(string DATAPATH, DTYPE **pLabel) {
    ifstream fin;

    if (DATAPATH == TEST_LABEL_FILE) {
        fin.open(TEST_LABEL_FILE, ios_base::binary);
    } else if (DATAPATH == TRAIN_LABEL_FILE) {
        fin.open(TRAIN_LABEL_FILE, ios_base::binary);
    }

    if (fin.is_open()) {
        int magicNumber = 0;
        int numOfLabels = 0;

        fin.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = ReverseInt(numOfLabels);

        for (int i = 0; i < numOfLabels; ++i) {
            pLabel[i] = new DTYPE[1];
            unsigned char data = 0;
            fin.read((char *)&data, sizeof(data));
            pLabel[i][0] = (DTYPE)data;
        }
    }
    fin.close();
}

template<typename DTYPE>
DTYPE** ReShapeData(OPTION pOption) {
    if (pOption == TESTIMAGE) {
        DTYPE **TestImage = new DTYPE *[NUMOFTESTDATA];
        IMAGE_Reader(TEST_IMAGE_FILE, TestImage);

        return TestImage;
    } else if (pOption == TESTLABEL) {
        DTYPE **TestLabel = new DTYPE *[NUMOFTESTDATA];
        LABEL_Reader(TEST_LABEL_FILE, TestLabel);

        return TestLabel;
    } else if (pOption == TRAINIMAGE) {
        DTYPE **TrainImage = new DTYPE *[NUMOFTRAINDATA];
        IMAGE_Reader(TRAIN_IMAGE_FILE, TrainImage);

        return TrainImage;
    } else if (pOption == TRAINLABEL) {
        DTYPE **TrainLabel = new DTYPE *[NUMOFTRAINDATA];
        LABEL_Reader(TRAIN_LABEL_FILE, TrainLabel);

        return TrainLabel;
    } else return NULL;
}

template<typename DTYPE>
MNISTDataSet<DTYPE>* CreateMNISTTrainDataSet(int bufferSize = 10, int batchSize = 100, int isGPUData = -1) {
    MNISTDataSet<DTYPE> *dataset = new MNISTDataSet<DTYPE>(TRUE, bufferSize, batchSize, isGPUData);

    dataset->SetTrainImage(ReShapeData<DTYPE>(TRAINIMAGE));
    dataset->SetTrainLabel(ReShapeData<DTYPE>(TRAINLABEL));

    return dataset;
}

template<typename DTYPE>
MNISTDataSet<DTYPE>* CreateMNISTTestDataSet(int bufferSize = 10, int batchSize = 100, int isGPUData = -1) {
    MNISTDataSet<DTYPE> *dataset = new MNISTDataSet<DTYPE>(FALSE, bufferSize, batchSize, isGPUData);

    dataset->SetTestImage(ReShapeData<DTYPE>(TESTIMAGE));
    dataset->SetTestLabel(ReShapeData<DTYPE>(TESTLABEL));

    return dataset;
}
