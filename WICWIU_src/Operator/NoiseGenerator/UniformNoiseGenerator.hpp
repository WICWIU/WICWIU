#ifndef UNIFORMNOISEGENERATOR_H_
#define UNOFORMNOISEGENERATOR_H_

#include <iostream>
#include <pthread.h>
#include <queue>
#include <semaphore.h>

#include "../../Common.h"
#include "../../Tensor.hpp" // to use IsUseTime
#include "../NoiseGenerator.hpp"

#define BUFF_SIZE 50
#define THREAD_NUM 4

template <typename DTYPE>
class UniformNoiseGenerator : public NoiseGenerator<DTYPE>
{
private:
    std::queue<Tensor<DTYPE>*>* m_aaQForNoise;

    float m_LowerLimit;
    float m_UpperLimit;
    IsUseTime m_Answer;

    // for thread
    pthread_t m_thread;

    sem_t m_full;
    sem_t m_empty;
    sem_t m_mutex;

    int m_isworking;

private:
    int Alloc()
    {
        m_aaQForNoise = new std::queue<Tensor<DTYPE>*>();
        return TRUE;
    }

public:
    UniformNoiseGenerator(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize,
                          int pColSize, float pLowerLimit, float pUpperLimit,
                          IsUseTime pAnswer = NoUseTime, std::string pName = "No Name")
        : NoiseGenerator<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize, pName)
    {
        m_LowerLimit = pLowerLimit;
        m_UpperLimit = pUpperLimit;
        m_Answer = pAnswer;

        m_isworking = 0;

        sem_init(&m_full, 0, 0);
        sem_init(&m_empty, 0, BUFF_SIZE);
        sem_init(&m_mutex, 0, 1);

        Alloc();
    }

    ~UniformNoiseGenerator(){};

    void StartProduce()
    {
        m_isworking = 1;

        for (int i = 0; i < THREAD_NUM; i++)
        {
            pthread_create(&m_thread, NULL, &UniformNoiseGenerator::ThreadFunc, (void*)this);
        }
    }

    void StopProduce()
    {
        m_isworking = 0;

        sem_post(&m_empty);
        sem_post(&m_full);

        pthread_join(m_thread, NULL);
    }

    static void* ThreadFunc(void* arg)
    {
        UniformNoiseGenerator<DTYPE>* generator = (UniformNoiseGenerator<DTYPE>*)arg;

        generator->GenerateNoise();
    }

    int GenerateNoise()
    {
        int m_timesize = this->GetResult()->GetDim(0);
        int m_batchsize = this->GetResult()->GetDim(1);
        int m_channelsize = this->GetResult()->GetDim(2);
        int m_rowsize = this->GetResult()->GetDim(3);
        int m_colsize = this->GetResult()->GetDim(4);

        do
        {
            Tensor<DTYPE>* temp = NULL;

            temp = Tensor<float>::Random_Uniform(m_timesize, m_batchsize, m_channelsize, m_rowsize,
                                                 m_colsize, m_LowerLimit, m_UpperLimit);

            sem_wait(&m_empty);
            sem_wait(&m_mutex);

            this->AddNoise2Buffer(temp);

            sem_post(&m_mutex);
            sem_post(&m_full);

        } while (m_isworking);
    }

    int AddNoise2Buffer(Tensor<DTYPE>* noise)
    {
        m_aaQForNoise->push(noise);

        return TRUE;
    }

    Tensor<DTYPE>* GetNoiseFromBuffer()
    {
        sem_wait(&m_full);
        sem_wait(&m_mutex);

        Tensor<DTYPE>* result = m_aaQForNoise->front();
        m_aaQForNoise->pop();

        sem_post(&m_mutex);
        sem_post(&m_empty);

        return result;
    }

    int ForwardPropagate(int pTime = 0){};
    int BackPropagate(int pTime = 0){};

private:
    int Alloc(Shape* pShape);
};
#endif // UNIFORMNOISEGENERATOR_H_
