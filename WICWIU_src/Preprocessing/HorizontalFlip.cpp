#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iterator>
#include <dirent.h>
#include <sys/types.h>

#include "../../WICWIU_src/Shape.hpp"
#include "../../WICWIU_src/ImageProcess.hpp"

class HorizentalFlip : public Transform
{
private:
    int m_heigth;
    int m_width;

public:
    HorizentalFlip(int heigth, int width) : m_heigth(heigth), m_width(width)
    {
    }

    HorizentalFlip(int size) : m_heigth(size), m_width(size)
    {
    }

    virtual ~HorizentalFlip() {}

    virtual void DoTransform(ImageWrapper &imgWrp)
    {
        unsigned char *imgBuf = imgWrp.imgBuf;
        Shape *imgShape = imgWrp.imgShape;

        int width = imgShape->GetDim(2);
        int height = imgShape->GetDim(1);
        int channel = imgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape *NewImgShape = new Shape(channel, m_heigth, m_width);

        //가로 플립.
        for (int h = 0; h < m_heigth; h++)
        {
            for (int w = 0; w < m_width; w++)
            {
                for (int ch = 0; ch < channel; ch++)
                {
                    newImgBuf[ch * height * width + h * width + w]

                        = imgBuf[ch * height * width + h * width + (width - w - 1)];
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