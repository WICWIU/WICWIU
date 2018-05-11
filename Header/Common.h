// C++ header
#include <iostream>
#include <stdexcept>
#include <exception>
#include <string>
#include <cmath>
#include <thread>

// C header
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

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
