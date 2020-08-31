#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <vector>

#include "../../WICWIU_src/ImageProcess.hpp"
#include "../../WICWIU_src/Shape.hpp"

class RandomCrop : public Transform
{
private:
    int m_heigth;
    int m_width;

public:
    RandomCrop(int heigth, int width) : m_heigth(heigth), m_width(width) {}

    RandomCrop(int size) : m_heigth(size), m_width(size) {}

    virtual ~RandomCrop() {}

    virtual void DoTransform(ImageWrapper& imgWrp)
    {
        unsigned char* imgBuf = imgWrp.imgBuf;
        Shape* imgShape = imgWrp.imgShape;

        int width = imgShape->GetDim(2);
        int height = imgShape->GetDim(1);
        int channel = imgShape->GetDim(0);

        unsigned char* newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape* NewImgShape = new Shape(channel, m_heigth, m_width);
        srand((unsigned)time(NULL) + (unsigned int)(intptr_t)NewImgShape);

        int limit_h = height - m_heigth;
        int limit_w = width - m_width;

        int start_h = rand() % limit_h;
        int start_w = rand() % limit_w;

        for (int h = 0; h < m_heigth; h++)
        {
            int oldh = start_h + h;

            for (int w = 0; w < m_width; w++)
            {
                int oldw = start_w + w;

                for (int ch = 0; ch < channel; ch++)
                {
                    newImgBuf[ch * m_heigth * m_width + h * m_width + w] =
                        imgBuf[ch * height * width + oldh * width + oldw];
                }
            }
        }

        delete[] imgBuf;
        imgBuf = NULL;
        delete imgShape;
        imgShape = NULL;

        imgWrp.imgBuf = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};