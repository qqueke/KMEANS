/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "kmeans.h"
#include <cuda.h>
#include <omp.h>

#define RANDOM_MAX 2147483647
//change this value to get different benchmarks for the system
#define THREADS_PER_BLOCK 128

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

__constant__ int device_nclusters;
__constant__ int device_npoints;
__constant__ int device_nfeatures;
__constant__ float DEVICE_FLT_MAX;

__global__ 
void cuda_find_nearest_point(float *feature, float *clusters,  float *new_centers, int *new_centers_len, int *membership, float *delta)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i >= device_npoints) return;
    
    *delta = 0.0f;
    __syncthreads();

    int j, k, index;
    float min_dist=DEVICE_FLT_MAX;

    for (j=0; j<device_nclusters; j++) {
        
        float dist = 0.0;

        for (k=0; k<device_nfeatures; k++) {
            dist += (feature[i * device_nfeatures + k]-clusters[j * device_nfeatures + k]) * (feature[i * device_nfeatures + k]-clusters[j * device_nfeatures + k]); 
        }

        if (dist < min_dist) {
            min_dist = dist;
            index = j;
        }
    }

    atomicAdd(&new_centers_len[index], 1);

    for (j=0; j<device_nfeatures; j++)          
        atomicAdd(&new_centers[index * device_nfeatures + j], feature[i * device_nfeatures + j]);

    if (membership[i] != index) atomicAdd(delta, 1.0f);
    
    membership[i] = index;

}


__global__ void replace_cluster_centers(float *clusters, float *new_centers, int * new_centers_len){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(tid_x >= device_nclusters * device_nfeatures) return;

    int row_index = tid_x / device_nfeatures;

    if(new_centers_len[row_index] > 0)
        clusters[tid_x] = new_centers[tid_x] / new_centers_len[row_index];

    new_centers[tid_x] = 0.0;

    __syncthreads();

    new_centers_len[row_index]=0;
}

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership,
                          int * iterations) /* out: [npoints] */
{

    int i, j, n=0;
    int *new_centers_len;   /* [nclusters]: no. of points in each cluster */
    float **clusters;       /* out: [nclusters][nfeatures] */
    float **new_centers;    /* [nclusters][nfeatures] */
    float *delta;
    float aux = FLT_MAX;
    float *features;

    struct timespec t1, t2;
    double timing;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *device_delta, *device_threshold;
    int *device_membership, *device_new_centers_len;
    float *device_feature, *device_clusters, *device_new_centers;

    printf("NFeatures: %d\n", nfeatures);
    printf("NClusters: %d\n", nclusters);
    printf("NPoints: %d\n", npoints);

    cudaMallocHost((void**)&features, npoints * nfeatures * sizeof(float));

    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    
    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    
    cudaMallocHost((void **)&delta, sizeof(float));
    cudaMallocHost((void**)&new_centers_len, nclusters * sizeof(int));
     
    cudaMalloc((void **)&device_delta, sizeof(float)); 
    cudaMalloc((void **)&device_threshold, sizeof(float));
    cudaMalloc((void **)&device_new_centers_len, sizeof(int) * nclusters);
    cudaMalloc((void **)&device_membership, sizeof(int) * npoints);
    cudaMalloc((void **)&device_feature, sizeof(float) * nfeatures * npoints);
    cudaMalloc((void **)&device_clusters, sizeof(float) * nfeatures * nclusters);
    cudaMalloc((void **)&device_new_centers, sizeof(float) * nfeatures * nclusters);

    clock_gettime(CLOCK_REALTIME, &t1);

    for (i=0; i<npoints; i++){
        membership[i] = -1;
        memcpy(&features[i * nfeatures], feature[i], nfeatures * sizeof(float));
    }

    *delta = 0.0;

    cudaMemcpyAsync(device_feature, features, sizeof(float) * npoints * nfeatures, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbolAsync(device_nfeatures, &nfeatures, sizeof(int));
    cudaMemcpyToSymbolAsync(device_nclusters, &nclusters, sizeof(int));
    cudaMemcpyToSymbolAsync(device_npoints, &npoints, sizeof(int));
    cudaMemcpyToSymbolAsync(DEVICE_FLT_MAX, &aux, sizeof(float));

    cudaMemcpyAsync(device_new_centers_len, new_centers_len, sizeof(float) * nclusters, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(device_membership, membership, sizeof(int) * npoints, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(device_delta, delta, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(device_threshold, &threshold, sizeof(float), cudaMemcpyHostToDevice);

    /*From here on maybe we can launch a kernel to do this instead of copying more stuff to the gpu*/

    for (i=1; i<nclusters; i++){
        clusters[i] = clusters[i-1] + nfeatures;
        new_centers[i] = new_centers[i-1] + nfeatures;
    }
    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
		n++;
    
        cudaMemcpyAsync(&(device_clusters[i * nfeatures]), clusters[i], sizeof(float) * nfeatures, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&(device_new_centers[i * nfeatures]), new_centers[i], sizeof(float) * nfeatures, cudaMemcpyHostToDevice);
    }
    
    dim3 numThreads(THREADS_PER_BLOCK, 1, 1);
    dim3 numBlocksA(ceil((float)npoints/(float)numThreads.x), 1, 1);
    dim3 numBlocksB(ceil((float)nclusters * (float) nfeatures / (float) numThreads.x), 1, 1);
    dim3 numBlocksC(ceil((float)npoints * (float) nfeatures /(float)numThreads.x), 1, 1);
    
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_REALTIME, &t2);

    timing = (((double)t2.tv_sec - t1.tv_sec) * 1000.0) + (((double)t2.tv_nsec - t1.tv_nsec) / 1000000.0);

    printf("Time for memory transfer: %f [ms]\n", timing);

    clock_gettime(CLOCK_REALTIME, &t1);
    
    do {
        *iterations = *iterations + 1;

        cudaEventRecord(start);
        cuda_find_nearest_point<<<numBlocksA, numThreads>>>(device_feature, device_clusters, device_new_centers, device_new_centers_len, device_membership, device_delta);
        cudaEventRecord(stop);
        
        cudaMemcpyAsync(delta, device_delta, sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(stop);
        
        replace_cluster_centers<<<numBlocksB, numThreads>>>(device_clusters, device_new_centers, device_new_centers_len);
        
        //cudaDeviceSynchronize();

    } while (*delta > threshold);

    clock_gettime(CLOCK_REALTIME, &t2);

    timing = (((double)t2.tv_sec - t1.tv_sec) * 1000.0) + (((double)t2.tv_nsec - t1.tv_nsec) / 1000000.0);

    printf("Time for the loop: %f [ms]\n", timing);

    for(i=0; i<nclusters; i++) 
        cudaMemcpyAsync(clusters[i], &(device_clusters[i * nfeatures]) , sizeof(float) * nfeatures, cudaMemcpyDeviceToHost);
    
    cudaFree(device_new_centers_len);
    cudaFree(device_feature);
    cudaFree(device_new_centers);
    cudaFree(device_delta);
    cudaFree(device_threshold);
    cudaFree(device_membership);

    free(new_centers[0]);
    free(new_centers);

    cudaFreeHost(new_centers_len);
    cudaFreeHost(features);
    cudaFreeHost(delta);

    cudaDeviceSynchronize();
    
    cudaFree(device_clusters);

    return clusters;
}



