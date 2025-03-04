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

static cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint16_t *d_icost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;

static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;

static uint8_t p1, p2;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;

static void free_memory() {
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_L0));
	CUDA_CHECK_RETURN(cudaFree(d_L1));
	CUDA_CHECK_RETURN(cudaFree(d_L2));
	CUDA_CHECK_RETURN(cudaFree(d_L3));
#if PATH_AGGREGATION == 8
	CUDA_CHECK_RETURN(cudaFree(d_L4));
	CUDA_CHECK_RETURN(cudaFree(d_L5));
	CUDA_CHECK_RETURN(cudaFree(d_L6));
	CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
	CUDA_CHECK_RETURN(cudaFree(d_cost));
	CUDA_CHECK_RETURN(cudaFree(d_icost));
}

#if PATH_AGGREGATION == 8
__global__ void addKernel(uint16_t* c, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const uint8_t* d_L7, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = (uint16_t) d_L0[i] + (uint16_t) d_L1[i] + (uint16_t) d_L2[i] + (uint16_t) d_L3[i] + (uint16_t) d_L4[i] + (uint16_t) d_L5[i] + (uint16_t) d_L6[i] + (uint16_t) d_L7[i];
    }
}
#else
__global__ void addKernel(uint16_t* c, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = (uint16_t) d_L0[i] + (uint16_t) d_L1[i] + (uint16_t) d_L2[i] + (uint16_t) d_L3[i];
    }
}
#endif

void init_aggregate_method(const uint8_t _p1, const uint8_t _p2) {
	// We are not using shared memory, use L1
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	//CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	// Create streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
	first_alloc = true;
	p1 = _p1;
	p2 = _p2;
    rows = 0;
    cols = 0;
}

void compute_aggregate_method(uint16_t *h_idsi, uint8_t *h_disparity, cost_t *left_ct, cost_t *right_ct, uint8_t *h_dsi, uint32_t h, uint32_t w, float *elapsed_time_ms) {
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
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_icost, sizeof(uint16_t)*size_cube_l));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
#if PATH_AGGREGATION == 8
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
#endif

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
	}

	debug_log("Copying ct and dsi to the GPU");
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_transform0, left_ct, sizeof(cost_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_transform1, right_ct, sizeof(cost_t)*size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_cost, h_dsi, sizeof(uint8_t)*size_cube_l, cudaMemcpyHostToDevice, stream1));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

	debug_log("Calling Left to Right");
	CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	debug_log("Calling Right to Left");
	CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	debug_log("Calling Up to Down");
	CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	debug_log("Calling Down to Up");
	CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

#if PATH_AGGREGATION == 8
	CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
#endif

	dim3 block_size;
	block_size.x = 32;

	dim3 grid_size;
	grid_size.x = (size_cube_l+1) / block_size.x;

//https://riptutorial.com/cuda/example/6820/sum-two-arrays-with-cuda
#if PATH_AGGREGATION == 8
    addKernel<<<grid_size, block_size, 0, stream1>>>(d_icost, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, d_L7, size_cube_l);
#else
    addKernel<<<grid_size, block_size, 0, stream1>>>(d_icost, d_L0, d_L1, d_L2, d_L3, size_cube_l);
#endif

	cudaEventRecord(stop, 0);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaEventElapsedTime(elapsed_time_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	debug_log("Copying final disparity to CPU");
	CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_idsi, d_icost, sizeof(uint16_t)*size_cube_l, cudaMemcpyDeviceToHost));
}

void finish_aggregate_method() {
	if(!first_alloc) {
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}
