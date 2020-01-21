#include "Common.h"

#include <stdio.h>
#include <stdarg.h>
#include <float.h>


#ifdef	_WINDOWS
	#include <windows.h>
#else	//	_WINDOWS
	#include <pthread.h>
	#include <sys/stat.h>
	#include <sys/types.h>
#endif	//	_WINDOWS

#include "Utils.hpp"

void AllocFeatureVector(int dim, int noSample, std::vector<float*> &vFeature)
{
    if(vFeature.size() > 0)
        DeleteFeatureVector(vFeature);

    try{
        vFeature.resize(noSample);
        for(int i = 0; i < noSample; i++)
            vFeature[i] = new float[dim];
    } catch(...){
        printf("Failed to allocate memory (dim = %d, noSample = %d) in %s (%s %d)\n", dim, noSample, __FUNCTION__, __FILE__, __LINE__);
        MyPause(__FUNCTION__);
        return;
    }
}

void DeleteFeatureVector(std::vector<float*> &vFeature)
{
    for(int i = 0; i < vFeature.size(); i++)
        delete[] vFeature[i];

    vFeature.resize(0);
}

#ifdef	MultiThread
#ifdef	_WINDOWS
	HANDLE LogMessageF_Mutex = NULL;
#else	//	_WINDOWS
	pthread_mutex_t LogMessageF_Mutex = PTHREAD_MUTEX_INITIALIZER;
#endif	//	_WINDOWS
#endif	//	MultiThread

int LogMessageF(const char *logFile, int bOverwrite, const char *format, ...)
// log formatted message into file
// usage : same with printf(const char *format, ¡¦)
{
#ifdef	MultiThread
#ifdef	_WINDOWS
	if(LogMessageF_Mutex == NULL)
		LogMessageF_Mutex = CreateMutex(NULL, FALSE, NULL);
	if(LogMessageF_Mutex == NULL){
		printf("Failed to create mutex, %s (%s, %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return -1;
	}

	while(LogMessageF_Mutex == NULL);
	WaitForSingleObject(LogMessageF_Mutex, INFINITE);
#else	//	_WINDOWS
	pthread_mutex_lock(&LogMessageF_Mutex);
#endif	//	_WINDOWS
#endif	//	MultiThread

	va_list vl;					// in varargs.h
	static char buffer[1024];
	FILE *fp = NULL;

	va_start(vl, format);
	vsprintf(buffer, format, vl);
	va_end(vl);

	fp = fopen(logFile, bOverwrite ? "w" : "a");
	if(fp == NULL)
		return FALSE;

	fprintf(fp, "%s", buffer);

	fclose(fp);
#ifdef	MultiThread
#ifdef	_WINDOWS
	ReleaseMutex(LogMessageF_Mutex);
#else	//	_WINDOWS
	pthread_mutex_unlock(&LogMessageF_Mutex);
#endif	//	_WINDOWS
#endif	//	MultiThread
	return TRUE;
}

int DisplayFeature(int dim, float *data, int width)
{
    printf("dim = %d\n", dim);
    float *p = data;

	dim = MIN(dim, 16);			// for debug

    for(int x = 0; x < dim; x++, p++){
        printf("%.2f ", *p);
        if(width > 0 && (x + 1) % width == 0)
            printf("\n");
    }

    printf("\n");

    return TRUE;
}

int LogFeature(const char *fileName, int bOverwrite, int dim, float *data, int width)
{
    FILE *fp = fopen(fileName, (bOverwrite ? "w" : "a"));
    if(fp == NULL){
        printf("Failed to open file %s in %s (%s %d)\n", fileName, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    fprintf(fp, "dim = %d\n", dim);
    float *p = data;
    for(int x = 0; x < dim; x++, p++){
        fprintf(fp, "%f ", *p);
        if(width > 0 && (x + 1) % width == 0)
            fprintf(fp, "\n");
    }

    fprintf(fp, "\n");

    return TRUE;
}

int DisplayImage(int width, int height, float *data)
{
    printf("size = %d x %d\n", width, height);
    float *p = data;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++, p++)
            printf("%c", (*p >= 0.5 ? 'O' : '.'));

        printf("\n");
    }

    return TRUE;
}

int LogImage(const char *fileName, int bOverwrite, int width, int height, float *data)
{
    FILE *fp = fopen(fileName, (bOverwrite ? "w" : "a"));
    if(fp == NULL){
        printf("Failed to open file %s in %s (%s %d)\n", fileName, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    fprintf(fp, "size = %d x %d\n", width, height);
    float *p = data;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++, p++)
            fprintf(fp, "%c ", (*p >= 0.5 ? 'O' : '.'));

        fprintf(fp, "\n");
    }

    return TRUE;
}

void MyPause(const char *message)
{
    if(message)
        printf("Press Enter to continue (%s)...", message);
    else
        printf("Press Enter to continue ...");
    
    fflush(stdout);
    getchar();
}

float GetSquareDistance(int dim, float *pVec1, float *pVec2)
{
    float dist = 0.F;

    for(int i = 0; i < dim; i++){
        float diff = pVec1[i] - pVec2[i];
        if(!(diff * diff >= -FLT_MAX &&  diff * diff <= FLT_MAX)){
            printf("abnormal: diff = %f, pVec1[%d] = %f, pVec2[%d] = %f", diff, i, pVec1[i], i, pVec2[i]);            
            printf("Press Enter to continue... in %s", __FUNCTION__);
            fflush(stdout);
            getchar();
        }
        dist += diff * diff;

        if(!(dist >= -FLT_MAX &&  dist <= FLT_MAX)){
            printf("abnormal: dist = %f diff = %f, pVec1[%d] = %f, pVec2[%d] = %f", dist, diff, i, pVec1[i], i, pVec2[i]);
            printf("Press Enter to continue... in %s", __FUNCTION__);
            fflush(stdout);
            getchar();
        }
    }

    return dist;
}
