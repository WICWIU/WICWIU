#ifndef COMMON_H_
#define COMMON_H_    value


// C++ header
#include <iostream>
#include <stdexcept>
#include <exception>
#include <string>
#include <cmath>
#include <zlib.h>
// #include <thread>

// C header
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

// Custom code
#include "Utils.hpp"

// define
#ifndef TRUE
    # define TRUE     1
    # define FALSE    0
#endif  // !TRUE

#ifdef __CUDNN__
    # include "cuda.h"
    # include "cudnn.h"
    # include "error_util.h"
#endif  // ifndef __CUDNN__

/*!
@brief 장치 사용 구분자, CPU 또는 GPU
@details 학습 시 사용하는 장치를 구분하기 위해 사용하는 열거형 변수, CPU 또는 GPU
*/
enum Device {
    CPU,
    GPU,
};

#endif  // COMMON_H_
