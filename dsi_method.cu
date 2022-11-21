/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2022 Luca Bartolomei.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include "aggregate_method.h"

static cudaStream_t stream1;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;


static void free_memory() {
	CUDA_CHECK_RETURN(cudaFree(d_im0));
	CUDA_CHECK_RETURN(cudaFree(d_im1));
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_cost));
}

void init_dsi_method() {
	// We are not using shared memory, use L1
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	// Create streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	first_alloc = true;
    rows = 0;
    cols = 0;
}

void compute_dsi_method(uint8_t *h_dsi,  uint8_t *left_ct, uint8_t *right_ct, uint8_t *left, uint8_t *right, uint32_t h, uint32_t w, float *elapsed_time_ms) {
	if(cols != w || rows != h) {
		debug_log("WARNING: cols or rows are different");
		if(!first_alloc) {
			debug_log("Freeing memory");
			free_memory();
		}
		first_alloc = false;
		cols = w;
		rows = h;
		size = rows*cols;
		size_cube_l = size*MAX_DISPARITY;
        
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube_l));
	}

	debug_log("Copying images to the GPU");
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

	debug_log("Calling CSCT");
	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	debug_log("Calling Hamming Distance");
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	cudaEventRecord(stop, 0);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaEventElapsedTime(elapsed_time_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	debug_log("Copying final dsi to CPU");
	CUDA_CHECK_RETURN(cudaMemcpy(h_dsi, d_cost, sizeof(uint8_t)*size_cube_l, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(left_ct, d_transform0, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(right_ct, d_transform1, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
}

void finish_dsi_method() {
	if(!first_alloc) {
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
	}
}
